# reference paper: https://arxiv.org/pdf/1609.05473.pdf
#
# original code from LantaoYu
# 
#



import numpy as np
#import tensorflow as tf

class Gen_Data_loader():
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size, seq_length, vocab_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [float(x) for x in line]
                positive_examples.append(parse_line)

        self.positive_examples = positive_examples
        self.np_positive_examples = np.array(self.positive_examples)
        self.num_batch = int(len(self.np_positive_examples) / self.batch_size) #original num_batch for each postive, negative file
        self.np_positive_examples = self.np_positive_examples[:self.num_batch * self.batch_size]
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [float(x) for x in line]
                if len(parse_line) == self.vocab_size:
                    negative_examples.append(parse_line)
        self.negative_examples = negative_examples
        self.np_negative_examples = np.array(self.negative_examples)
        self.np_negative_examples = self.np_negative_examples[:self.num_batch * self.batch_size * self.seq_length]
        self.np_negative_examples = np.array(np.split(self.np_negative_examples, self.num_batch*self.batch_size, 0))

        total_vocab = []
        for seq in self.np_positive_examples:
            seq_vocab=[]
            for idx in seq:
                vocab = [0] * self.vocab_size
                vocab[int(idx)] = 1
                seq_vocab.append(vocab)
            total_vocab.append(seq_vocab)
        self.np_positive_examples = np.array(total_vocab, dtype=float)
        # num_batch*batch_size, seq, vocab
        self.sentences = np.concatenate([self.np_positive_examples , self.np_negative_examples], 0)

        self.positive_labels = np.array([[0, 1] for _ in self.np_positive_examples])
        self.negative_labels = np.array([[1, 0] for _ in self.np_negative_examples])
        self.labels = np.concatenate([self.positive_labels , self.negative_labels], 0)
        #[num_batch*batch_size*2, 2]

        shuffle_indices = np.random.permutation(np.arange(len(self.labels))) #self.num_batch * self.batch_size
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.sentences_batches = np.split(self.sentences, self.num_batch*2, 0)
        self.labels_batches = np.split(self.labels, self.num_batch*2, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch*2
        return ret

    def reset_pointer(self):
        self.pointer = 0

