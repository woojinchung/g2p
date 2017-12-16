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
import models.encoder_decoder as enc_dec



def time_since(since):
    now = time.time()
    s = now - since
    h = s // 3600
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d:%d' % (h, m, s)


START_TIME = time.time()

class EncoderDecoderWithAttention(enc_dec.EncoderDecoder):
    def __init__(self, hidden_size, embedding_size, output_size, num_layers, out_seq_len, gpu):
        super(EncoderDecoderWithAttention, self).__init__(hidden_size, embedding_size, output_size, num_layers, out_seq_len, gpu)
        self.attention = nn.Linear(3*hidden_size + output_size, 1)
        self.gpu = gpu

    def forward_training(self, input, labels):
        batch_size = len(input)
        input = torch.transpose(input, 0, 1)
        in_seq_len = input.size()[0]
        hiddens, _ = self.encoder(input)
        (h, c) = self.init_hidden_decoder(batch_size)[0]
        if self.gpu:
            h = h.cuda()
            c = c.cuda()
        outputs = [labels[0]]
        for i, y in enumerate(labels[0:-1]):
            # get encoding using attention
            if self.gpu:
                y = y.cuda()

            tmp = torch.cat([
                y.expand(in_seq_len, batch_size, self.output_size),
                h.expand(in_seq_len, batch_size, self.hidden_size),
                hiddens], 2)
            alphas = self.attention(tmp)
            alphas = F.softmax(alphas, dim=0)
            alphas = alphas.expand(-1, -1, 2 * self.hidden_size)
            encoding = torch.sum(torch.mul(hiddens, alphas), 0)

            tmp = torch.cat([encoding, y], 1)
            h, c = self.decoder(tmp, (h, c))
            output = F.softmax(self.h2o(h), dim=1)
            outputs.append(output)
        return outputs

    def forward_eval(self, input):
        batch_size = len(input)
        input = torch.transpose(input, 0, 1)
        in_seq_len = input.size()[0]
        hiddens, _ = self.encoder(input)
        (h, c) = self.init_hidden_decoder(batch_size)[0]

        #the first output needs to fabricated as the start symbol
        outputs = []
        y = Variable(torch.zeros(batch_size, self.output_size))
        for i in range(batch_size):
            y[i, 0] = 1
        outputs.append(y)
        for i in range(self.out_seq_len-1):
            # get encoding using attention
            tmp = torch.cat([
                y.expand(in_seq_len, batch_size, self.output_size),
                h.expand(in_seq_len, batch_size, self.hidden_size),
                hiddens], 2)
            alphas = self.attention(tmp)
            alphas = F.softmax(alphas, dim=0)
            alphas = alphas.expand(-1, -1, 2 * self.hidden_size)
            encoding = torch.sum(torch.mul(hiddens, alphas), 0)

            tmp = torch.cat([encoding, y], 1)
            h, c = self.decoder(tmp, (h, c))
            output = F.softmax(self.h2o(h), dim=1)
            outputs.append(output)
            y = output
        return outputs
