import torch
import random
import time
import torch.nn as nn
import gflags
from torch.autograd import Variable
from models import model_trainer
import torch.nn.functional as F
import math

# EVALUATE_EVERY = 1000
#
# LOGS = open("logs/rnn-logs", "a")
# OUTPUT_PATH = "models/rnn_classifier"



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
    def __init__(self, hidden_size, embedding_size, output_size, num_layers):
        super(EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.LSTMCell(2*hidden_size + output_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward_training(self, input, labels):
        batch_size = len(input)
        encoding, _ = self.encoder(input)
        encoding = F.sigmoid(encoding)
        encoding = torch.transpose(encoding, 0, 1)
        (h, c) = self.init_hidden_decoder(batch_size)[0]
        outputs = []
        for i, y in enumerate(labels):
            tmp = torch.cat([encoding[i], y], 1)
            h, c = self.decoder(tmp, (h, c))
            output = F.softmax(self.h2o(h))
            outputs.append(output)
        return outputs
        # reshape output
        # tmp = Variable(torch.zeros(len(outputs), batch_size, self.output_size))
        # for i in range(len(outputs)):
        #     tmp[i] = outputs[i]
        # # tmp = torch.transpose(tmp, 0, 1)
        # return tmp

        # trim output
        # tmp = Variable(torch.zeros(len(outputs), batch_size, self.output_size))
        # for i in range(len(outputs)):
        #     tmp[i] = outputs[i]
        # tmp = torch.transpose(tmp, 0, 1)
        # trimmed_outputs = []
        # for i, outputs in enumerate(tmp):
        #     trimmed_outputs.append(outputs[:lengths[i], :].cpu())
        # return trimmed_outputs

    def forward_eval(self, input):
        batch_size = len(input)
        encoding, _ = self.encoder(input)
        encoding = F.sigmoid(encoding)
        encoding = torch.transpose(encoding, 0, 1)
        (h, c) = self.init_hidden_decoder(batch_size)[0]
        outputs = []
        y = Variable(torch.zeros(batch_size, self.output_size))
        for i in range(batch_size):
            y[i, 0] = 1
        for i in range(input.size()[1]):
            tmp = torch.cat([encoding[i], y], 1)
            h, c = self.decoder(tmp, (h, c))
            output = F.softmax(self.h2o(h))
            outputs.append(output)
            y = output
        # return outputs
        # reshape output
        tmp = Variable(torch.zeros(len(outputs), batch_size, self.output_size))
        for i in range(len(outputs)):
            tmp[i] = outputs[i]
        # tmp = torch.transpose(tmp, 0, 1)
        return tmp

        # trim output
        # tmp = Variable(torch.zeros(len(outputs), batch_size, self.output_size))
        # for i in range(len(outputs)):
        #     tmp[i] = outputs[i]
        # tmp = torch.transpose(tmp, 0, 1)
        # trimmed_outputs = []
        # for i, outputs in enumerate(tmp):
        #     trimmed_outputs.append(outputs[:lengths[i], :].cpu())
        #
        # return trimmed_outputs

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
        super(EDTrainer, self).__init__(FLAGS, model, optimizer)

    def run_batch(self, source_batch, labels_list, backprop):
        if self.FLAGS.gpu:
            input = source_batch[0].cuda()
        else:
            input = source_batch[0].cpu()
        one_hot_labels, by_index_labels = self.pad_labels(labels_list, self.model.output_size)
        if backprop:
            outputs_list = self.model.forward_training(input, one_hot_labels)
            # outputs_list = self.model.forward_eval(input)
        else:
            outputs_list = self.model.forward_eval(input)

        #reshape labels for loss
        tmp = Variable(torch.zeros(len(outputs_list), len(labels_list), self.model.output_size))
        # for i in range(len(labels)):
        #     tmp[i] = labels[i]
        loss, correct, preds = self.get_metrics(outputs_list, [x.long() for x in by_index_labels])

        if backprop:
            self.backprop(loss)
        if self.FLAGS.gpu:
            loss = loss.cpu()

        return loss, correct, preds

    def pad_labels(self, labels, output_size):
        max_seq_length = max([len(x) for x in labels])
        one_hot = [Variable(torch.zeros(len(labels), output_size)) for _ in range(max_seq_length)]
        by_index = [Variable(torch.zeros(len(labels))) for _ in range(max_seq_length)]
        for i, source in enumerate(labels):
            for j in range(len(source)):
                index = source[j].data[0]
                one_hot[j][i, index] = 1
                by_index[j][i] = index
        return one_hot, by_index


    def to_string(self):
        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            self.model.to_string() + \
            "learning rate\t\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "experiment name\t\t\t" + self.FLAGS.experiment_name

