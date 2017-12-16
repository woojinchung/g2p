import torch
import random
import time
import torch.nn as nn
import gflags
from torch.autograd import Variable
from models import model_trainer
import torch.nn.functional as F
import math
import utils.enc_dec_data_utils as cdu



def time_since(since):
    now = time.time()
    s = now - since
    h = s // 3600
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d:%d' % (h, m, s)


START_TIME = time.time()

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, num_layers, out_seq_len, gpu=False):
        super(EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.out_seq_len = out_seq_len
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.LSTMCell(2*hidden_size + output_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)


    def forward_training(self, input, labels):
        batch_size = len(input)
        input = torch.transpose(input, 0, 1)
        hiddens, (h_n, _) = self.encoder(input)

        encoding = hiddens[-1]

        # make encoding by max pooling hidden state of each layer at end of sequence
        # encoding = F.max_pool1d(torch.transpose(h_n, 0, 2), h_n.size()[0])
        # encoding = torch.transpose(encoding.squeeze(), 0, 1)
        (h, c) = self.init_hidden_decoder(batch_size)[0]
        outputs = [labels[0]]
        for i, y in enumerate(labels[0:-1]):
            tmp = torch.cat([encoding, y], 1)        #why do I do I take encoding[i]???
            h, c = self.decoder(tmp, (h, c))
            output = F.softmax(self.h2o(h),1)
            outputs.append(output)
        return outputs

    def forward_eval(self, input):
        batch_size = len(input)
        input = torch.transpose(input, 0, 1)
        _, (h_n, _) = self.encoder(input)
        hiddens, (h_n, _) = self.encoder(input)
        encoding = hiddens[-1]
        # encoding = F.max_pool1d(torch.transpose(h_n, 0, 2), h_n.size()[0])
        # enconing = torch.transpose(encoding.squeeze(), 0, 1)
        (h, c) = self.init_hidden_decoder(batch_size)[0]
        outputs = []
        y = Variable(torch.zeros(batch_size, self.output_size))
        for i in range(batch_size):
            y[i, 0] = 1
        outputs.append(y)
        for i in range(self.out_seq_len-1):
            tmp = torch.cat([encoding, y], 1)
            h, c = self.decoder(tmp, (h, c))
            output = F.softmax(self.h2o(h), 1)
            outputs.append(output)
            y = output
        return outputs


    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def init_hidden_decoder(self, batch_size):
        hidden_states = []
        for i in range(self.num_layers + 1):
            hidden_states.append((Variable(torch.zeros(batch_size, self.hidden_size)),
                                  Variable(torch.zeros(batch_size, self.hidden_size))))
        return hidden_states

    def to_string(self):
        return "input size\t\t" + str(self.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.hidden_size) + "\n" + \
            "reduction size\t\t" + str(self.output_size) + "\n" + \
            "num layers\t\t" + str(self.num_layers) + "\n"


class EDTrainer(model_trainer.ModelTrainer):
    def __init__(self,
                 FLAGS,
                 model,
                 optimizer):
        self.FLAGS = FLAGS
        self.model = model
        self.optimizer = optimizer
        self.dm = cdu.DataManager(self.FLAGS.data_dir, self.FLAGS.embedding_size)
        self.loss = torch.nn.CrossEntropyLoss()
        self.OUTPUT_PATH = FLAGS.ckpt_path + "/" + FLAGS.experiment_name + ".ckpt"
        self.LOGS_PATH = FLAGS.log_path + "LOGS-" + FLAGS.experiment_name
        self.OUT_LOGS_PATH = FLAGS.log_path + "OUTPUTS-" + FLAGS.experiment_name
        self.LOGS = open(self.LOGS_PATH, "a")
        self.OUT_LOGS = open(self.OUT_LOGS_PATH, "a")

    def run_batch(self, source_batch, labels_list, backprop):
        if self.FLAGS.gpu:
            input = source_batch[0].cuda()
        else:
            input = source_batch[0].cpu()
        one_hot_labels, by_idx_labels = self.pad_labels(labels_list, self.model.output_size)
        if backprop:
            outputs_list = self.model.forward_training(input, one_hot_labels)
        else:
            outputs_list = self.model.forward_eval(input)
        #reshape labels for loss
        loss, correct, guess, wcorrect, wguess = self.get_metrics(outputs_list, [x.long() for x in by_idx_labels])
        if backprop:
            self.backprop(loss)
        if self.FLAGS.gpu:
            loss = loss.cpu()
        return loss, correct, guess, wcorrect, wguess

    def pad_labels(self, labels, output_size):
        max_seq_length = max([len(x) for x in labels])
        one_hot = [Variable(torch.zeros(len(labels), output_size)) for _ in range(max_seq_length)]
        by_idx = [Variable(torch.zeros(len(labels))) for _ in range(max_seq_length)]
        for i, source in enumerate(labels):
            for j in range(len(source)):
                index = source[j].data[0]
                one_hot[j][i, index] = 1
                by_idx[j][i] = index
        return one_hot, by_idx

    def get_metrics(self, outputs_list, labels_list):
        if self.FLAGS.gpu:
            outputs_list = [output.cuda() for output in outputs_list]
            labels_list = [label.cuda() for label in labels_list]

        loss = 0

        correct_list = []
        trivial = 0
        for i in range(len(outputs_list)):
            # calculate loss
            logits = torch.log(outputs_list[i])
            loss += self.loss(logits, labels_list[i])

            # calculate class accuracy
            pred = logits.data.max(1)[1].cpu()  # get the index of the max log-probability
            correct = pred.eq(labels_list[i].cpu().data)
            for p, l in zip(pred, labels_list[i].cpu().data):
                if p == l and (p == 0 or p == 1):
                    trivial += 1
            correct_list.append(correct)

        correct_list = sum(correct_list)
        total_correct = sum(correct_list) - trivial
        total_guess = len(labels_list) * len(labels_list[0]) - trivial
        total_wguess = self.FLAGS.batch_size
        total_wcorrect = 0
        for i in correct_list:
            if i == len(labels_list):
                total_wcorrect += 1

        return loss, total_correct, total_guess, total_wcorrect, total_wguess




    def to_string(self):
        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            self.model.to_string() + \
            "learning rate\t\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "experiment name\t\t\t" + self.FLAGS.experiment_name

