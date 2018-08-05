"""
    That module regroups the objects and functions associated with data
    preprocessing
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_VECTORS_DIR = os.path.join(PKG_DIR, 'word_vectors')
DATASET_DIR = os.path.join(PKG_DIR, 'datasets')


class Vocab():
    '''
        The Vocab class is used to store the vocabulary found in the
        training data as well as the word embeddings.
    '''

    def __init__(self, word_vectors_file='glove.6B.50d.txt'):
        self.word_vectors_file = word_vectors_file
        self.word_to_index = {}
        self.index_to_word = {}
        self.total_words = 0 # total number of words parsed
        self.word_freq = defaultdict(int)
        self.padding = '<pad>'
        self.unknown = '<unk>'
        self._add_word(self.padding, count=0)
        self._add_word(self.unknown, count=0)
        self.embedding_matrix = None

    def _add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, texts):
        # add all words from the word_vectors file
        glove_file = os.path.join(WORD_VECTORS_DIR, self.word_vectors_file)
        for line in open(glove_file, 'r'):
            word = line.split(' ')[0]
            self._add_word(word, count=0)

        # add words from texts
        for text in texts:
            text = text.split()
            for word in text:
                self._add_word(word)
        self.total_words = sum(self.word_freq.values())
        print('{} total words parsed and {} unique words'.format(self.total_words, len(self.word_freq)))

    def encode(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index[self.unknown]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)

    def build_embedding_matrix(self):
        glove_file = os.path.join(WORD_VECTORS_DIR, self.word_vectors_file)
        with open(glove_file) as f:
            # get dimensions of word vectors
            dim = len(f.readline().split())-1
        self.embedding_matrix = np.zeros((self.__len__(), dim))
        for line in open(glove_file, 'r'):
            dat = line.split(' ')
            word = dat[0]
            if word in self.word_to_index:
                embedding = np.array(dat[1:], dtype='float32')
                self.embedding_matrix[self.word_to_index[word]] = embedding


def process_dataset(frac):
    '''
        Preprocess the input file, split the data randomly into training and
        validation set

       Args:
            frac (float): fraction of the whole data to be held out as
            validation data

        Returns:
            dict mapping original labels to int
    '''
    df = ( pd.read_csv(os.path.join(DATASET_DIR, 'dataset.tsv'), sep='\t', header=None)
             .rename(columns={0:'label', 1:'text'}) )

    # process labels
    df['label'] = df['label'].astype(str)
    labels = df.label.unique().tolist()
    labels.sort()

    # map labels to ints
    label_to_ind = {label:i for i, label in enumerate(labels)}
    df['label'] = df['label'].apply(lambda x: label_to_ind[x])

    # split dataset to train/valid according to frac
    df['train'] = 1
    valid_inds = np.random.choice(len(df), int(frac*len(df)), replace=False)
    df.iloc[valid_inds, df.columns.get_loc('train')] = 0

    # write the 2 data files
    df.loc[df.train==1,['label', 'text']].to_csv(os.path.join(DATASET_DIR,
        'dataset_train.tsv'), header=False, index=False, sep='\t')
    df.loc[df.train==0,['label', 'text']].to_csv(os.path.join(DATASET_DIR,
        'dataset_valid.tsv'), header=False, index=False, sep='\t')
    return label_to_ind


def clean_text(text):
    '''
        Perform text cleaning
    '''
    _text = text.lower()

    # separate punctuation from words
    _text = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", _text)

    # remove all but those characters (alphanumeric, punctuation, apostrophe)
    _text = re.sub('[^a-z0-9.,!?\-\' ]+', '', _text)
    _text = ( _text.replace('.', ' . ')
                   .replace('!', ' ! ')
                   .replace('?', ' ? ')
                   .replace(',', ' , ')
                   .replace(';', ' ; ')
                   .replace(':', ' : '))
    return _text


def get_words_dataset():
    '''
        Iterator that returns the words in the training set file
    '''
    dataset_file = os.path.join(DATASET_DIR, 'dataset_train.tsv')
    for line in open(dataset_file, 'r'):
        _, text = line.split('\t')
        text = clean_text(text).split()
        for word in text:
            yield word


def pad_text(vocab, text, target_len):
    if len(text) > target_len:
        _text = text[0:target_len]
    else:
        _fix = [vocab.encode(vocab.padding)]*(target_len-len(text))
        _text = text + _fix
    return _text


def encode_text(vocab, text):
    _cleaned_text = clean_text(text).split()
    _encoded_text = [vocab.encode(word) for word in _cleaned_text]
    return _encoded_text


def get_texts_dataset(vocab, target_len, split):
    dataset_file = os.path.join(DATASET_DIR, 'dataset_{}.tsv'.format(split))
    for line in open(dataset_file):
        label, text = line.split('\t')
        label = int(label)
        text = encode_text(vocab, text)
        length = min(len(text), target_len)
        if length == 0:
            length = 1
            text = [vocab.encode(vocab.unknown)]
        text = pad_text(vocab, text, target_len)
        text = np.array(text, dtype=np.int32)
        yield text, length, label


def load_dataset(vocab, target_len, split):
    data = list(get_texts_dataset(vocab, target_len, split))
    ( text, lengths, y ) = zip(*data)

    # arange in numpy arrays
    text = np.vstack(text)
    lengths = np.array(lengths, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    return text, lengths, y


def batch_iterator(texts, lengths, y=None, batch_size=32, shuffle=True):
    '''
        Iterator to generate batches of data
    '''
    N = len(texts)
    if shuffle:
        inds = np.random.permutation(N)
    else:
        inds = np.arange(N)
    _texts = texts[inds,:]
    _lengths = lengths[inds]
    _y = y[inds] if np.any(y) else None
    n_batches = int(np.ceil(N / batch_size))
    for i in range(n_batches):
        texts_b = _texts[i*batch_size:min((i+1)*batch_size, N), :]
        lengths_b = _lengths[i*batch_size:min((i+1)*batch_size, N)]
        y_b = None
        if np.any(_y):
            y_b = _y[i*batch_size:min((i+1)*batch_size, N)]
        yield texts_b, lengths_b, y_b
