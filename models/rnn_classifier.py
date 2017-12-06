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

class Classifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, reduction_size, num_layers, biLSTM):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.reduction_size = reduction_size
        self.ih2h = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=biLSTM)
        self.h2r = nn.Linear(hidden_size * 2 if biLSTM else hidden_size, reduction_size)

    def forward(self, input, input_lengths):
        batch_size = len(input)
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        packed_outputs, _ = self.ih2h(packed_inputs)
        outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        tmp = torch.chunk(outputs, batch_size, 0)
        tmp = torch.cat(tmp, 1).squeeze()
        tmp = F.sigmoid(self.h2r(tmp))
        tmp = tmp.view(batch_size, -1, self.reduction_size)

        # trim output
        trimmed_outputs = []
        for i, output in enumerate(tmp):
            trimmed_outputs.append(output[:lengths[i], :].cpu())

        return trimmed_outputs

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def to_string(self):
        return "input size\t\t" + str(self.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.hidden_size) + "\n" + \
            "reduction size\t\t" + str(self.reduction_size) + "\n" + \
            "num layers\t\t" + str(self.num_layers) + "\n"


class DeepClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, reduction_size, num_layers):
        super(DeepClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.reduction_size = reduction_size
        self.lstm = nn.LSTM(embedding_size, hidden_size/2, num_layers=num_layers, bidirectional=False)
        self.bilstm = nn.LSTM(embedding_size, hidden_size/4, num_layers=num_layers, bidirectional=True)
        self.deeplstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, bidirectional=False)
        self.h2r = nn.Linear(hidden_size, reduction_size)

    def forward(self, input, input_lengths):
        batch_size = len(input)
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)

        packed_lstm_outputs, _ = self.lstm(packed_inputs)
        packed_bilstm_outputs, _ = self.bilstm(packed_inputs)

        # merge two intermediate outputs
        lstm_outputs, lstm_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_outputs, batch_first=True)
        bilstm_outputs, bilstm_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_bilstm_outputs, batch_first=True)
        merged = torch.cat((lstm_outputs, bilstm_outputs), 2)
        packed_merged = torch.nn.utils.rnn.pack_padded_sequence(merged, lstm_lengths, batch_first=True)

        packed_deeplstm_outputs, deeplstm_lengths = self.deeplstm(packed_merged)
        deeplstm_outputs, deeplstm_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_deeplstm_outputs, batch_first=True)

        tmp = torch.chunk(deeplstm_outputs, batch_size, 0)
        tmp = torch.cat(tmp, 1).squeeze()
        tmp = F.sigmoid(self.h2r(tmp))
        tmp = tmp.view(batch_size, -1, self.reduction_size)

        # trim output
        trimmed_outputs = []
        for i, output in enumerate(tmp):
            trimmed_outputs.append(output[:deeplstm_lengths[i], :].cpu())

        return trimmed_outputs

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def to_string(self):
        return "input size\t\t" + str(self.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.hidden_size) + "\n" + \
            "reduction size\t\t" + str(self.reduction_size) + "\n" + \
            "num layers\t\t" + str(self.num_layers) + "\n"

class RNNTrainer(model_trainer.ModelTrainer):
    def __init__(self,
                 FLAGS,
                 model,
                 optimizer):
        self.FLAGS = FLAGS
        super(RNNTrainer, self).__init__(FLAGS, model, optimizer)


    def to_string(self):
        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            self.model.to_string() + \
            "learning rate\t\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "experiment name\t\t\t" + self.FLAGS.experiment_name


#============= EXPERIMENT ================



def random_experiment():
    h_size = int(math.floor(math.pow(random.uniform(10, 32), 2)))           # [100, 1024], quadratic distribution
    num_layers = random.randint(1, 5)
    reduction_size = int(math.floor(math.pow(random.uniform(7, 18), 2)))    # [49, 324], quadratic distribution
    lr = math.pow(.1, random.uniform(3, 4.5))                               # [.001, 3E-5], logarithmic distribution
    cl = Classifier(hidden_size=h_size, embedding_size=300, num_layers=num_layers, reduction_size=reduction_size)
    clt = RNNTrainer('/scratch/asw462/data/discriminator/',
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=True,
                     learning_rate=lr)
    clt.run()

def random_experiment_pooling(data):
    h_size = int(math.floor(math.pow(random.uniform(15, 40), 2)))  # [225, 1600], quadratic distribution
    num_layers = random.randint(1, 3)
    lr = math.pow(.1, random.uniform(3, 4.5))  # [.001, 3E-5], logarithmic distribution
    cl = ClassifierPooling(hidden_size=h_size, embedding_size=300, num_layers=num_layers)
    clt = RNNTrainer(data,
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=True,
                     learning_rate=lr)
    clt.run()

def random_local_experiment_pooling():
    h_size = int(math.floor(math.pow(random.uniform(10, 32), 2)))  # [100, 1024], quadratic distribution
    num_layers = random.randint(1, 5)
    lr = math.pow(.1, random.uniform(3.5, 5.5))  # [.001, 3E-5], logarithmic distribution
    cl = ClassifierPooling(hidden_size=h_size, embedding_size=300, num_layers=num_layers)
    clt = RNNTrainer('../data/discriminator/',
                     '../data/bnc-30/embeddings_20000.txt',
                    '../data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=False,
                     learning_rate=lr)
    clt.run()

def resume_experiment(model_path, h_size, num_layers, reduction_size, lr):
    cl = Classifier(hidden_size=h_size, embedding_size=300, num_layers=num_layers, reduction_size=reduction_size)
    cl.load_state_dict(torch.load(model_path))
    clt = RNNTrainer('/scratch/asw462/data/discriminator/',
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=False,
                     learning_rate=lr)
    clt.run()

def resume_experiment_pooling(model_path, h_size, num_layers, lr, data):
    cl = ClassifierPooling(hidden_size=h_size, embedding_size=300, num_layers=num_layers)
    cl.load_state_dict(torch.load(model_path))
    clt = RNNTrainer(data,
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=False,
                     learning_rate=lr)
    clt.run()






        # random_local_experiment_pooling()