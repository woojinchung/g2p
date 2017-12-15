import torch
import random
import torch.utils.data
import subprocess
import math
from functools import reduce
from torch.autograd import Variable
import io


class CorpusEpoch:
    def __init__(self, data_pairs, data_manager, batch_size=64):
        self.batch_size = batch_size
        self.data_manager = data_manager
        self.n_lines = len(data_pairs[0])
        self.n_batches = self.n_lines / self.batch_size

        # shuffle dataset
        z = list(zip(data_pairs[0], data_pairs[1]))
        random.shuffle(z)
        source, target = zip(*z)
        self.data_pairs = tuple((list(source), list(target)))

    def get_new_batch(self):
        source_batch_list = []
        labels_list = []

        for _ in range(min(self.batch_size, len(self.data_pairs[0]))):
            source = self.data_pairs[0].pop()
            target = self.data_pairs[1].pop()

            source_emb = torch.index_select(self.data_manager.emb, 0, torch.LongTensor(source))
            source_batch_list.append(source_emb)

            labels = Variable(torch.LongTensor(target))
            labels_list.append(labels)

        # sort batch by length in a descending order (required for variable length RNN)
        sorted_idx_list = sorted(range(len(source_batch_list)), key=lambda k: len(source_batch_list[k]), reverse=True)

        source_batch_list[:] = [source_batch_list[i] for i in sorted_idx_list]
        labels_list[:] = [labels_list[i] for i in sorted_idx_list]

        # create a Variable of size [batch_size x max_seq_length x emb_dim]
        batch_size = len(source_batch_list)
        max_seq_length = len(source_batch_list[0])
        emb_dim = len(source_batch_list[0][0])

        source_batch = Variable(torch.zeros(batch_size, max_seq_length, emb_dim))

        lengths = []
        for i, source in enumerate(source_batch_list):
            # pack_padded_sequence() requires the length of each sequence as input
            length = len(source)
            lengths.append(length)

            # fill in the Variable
            for j in range(length):
                source_batch[i, j] = source_batch_list[i][j]

        has_next = len(self.data_pairs[0]) > 0

        return tuple((source_batch, lengths)), labels_list, has_next


class DataManager:
    def __init__(self, corpus_path, embedding_size, unked=False):
        self.embedding_size = embedding_size
        self.training, self.valid, self.test = corpus_path + "/train_set.txt", corpus_path + "/dev_set.txt", corpus_path + "/test_set.txt"
        self.input_lang = Lang('gra')
        self.input_lang.addsymbol('EOS')
        self.input_lang.addsymbol('PAD')
        self.output_lang = Lang('pho')
        self.output_lang.addsymbol('PAD')
        self.training_pairs = None
        self.valid_pairs = None
        self.test_pairs = None
        self.emb = None

        self.preprocess_data()
        self.build_embedding()

    def build_embedding(self):
        i = torch.LongTensor([range(self.input_lang.n_symbols), range(self.input_lang.n_symbols)])
        v  = torch.ones(self.input_lang.n_symbols)
        self.emb = torch.sparse.FloatTensor(i, v, torch.Size([self.input_lang.n_symbols, self.input_lang.n_symbols])).to_dense()

    def preprocess_data(self):
        train_pairs = self.read_dict(self.training)
        dev_pairs = self.read_dict(self.valid)
        test_pairs = self.read_dict(self.test)

        self.training_pairs = self.get_idx_list(train_pairs)
        self.valid_pairs = self.get_idx_list(dev_pairs)
        self.test_pairs = self.get_idx_list(test_pairs)

    def get_idx_list(self, pairs):
        #pairs = self.filter_pair(pairs)
        print('read %s lexical items' % len(pairs))

        source_idx_list = []
        target_idx_list = []

        for pair in pairs:
            source_idx = []
            target_idx = []

            pair[0] = list(pair[0])
            pair[1] = pair[1].split(' ')

            source_length = len(pair[0])
            target_length = len(pair[1])

            # pad source
            pair[0].append("EOS")
            for _ in range(target_length - 1):
                pair[0].append("PAD")

            # pad target
            tmp = ["PAD" for _ in range(source_length)]
            pair[1] = tmp + pair[1]

            for i in range(len(pair[0])):
                source_idx.append(self.input_lang.symbol2index(pair[0][i]))
                target_idx.append(self.output_lang.symbol2index(pair[1][i]))

            source_idx_list.append(source_idx)
            target_idx_list.append(target_idx)

        return (source_idx_list, target_idx_list)

    def filter_pair(self, p):
        #specifying how long we want out symbols to be (shorter is useful for trying with a smaller subset?)
        #I just set a random large number for including everything
        MAX_LENGTH = 30

        return len(list(p[0])) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH 

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    def read_dict(self, file):
        print("Reading lines...")

        # Read the file and split into lines
        lines = io.open(file, encoding='latin-1').readlines()
        lines = [x.replace('\n', '') for x in lines if x[0] != ';']

        # Split every line into pairs and normalize
        pairs = [[s for s in l.split('  ')] for l in lines]

        for pair in pairs:
            self.input_lang.addword(pair[0], '')
            self.output_lang.addword(pair[1], ' ')

        return pairs


class Lang:
    def __init__(self, name):
        self.name = name
        self._symbol2index = {}
        self._index2symbol = {}
        self.n_symbols = 0

    def addword(self, word, s):
        if s == '':
            for symbol in list(word):
                self.addsymbol(symbol)
        else:
            for symbol in word.split(s):
                self.addsymbol(symbol)

    def addsymbol(self, symbol):
        if symbol not in self._symbol2index:
            self._symbol2index[symbol] = self.n_symbols
            self._index2symbol[self.n_symbols] = symbol
            self.n_symbols += 1
    
    def symbol2index(self, symbol):
        return self._symbol2index[symbol]

    def index2symbol(self, idx):
        return self._index2symbol[idx]
