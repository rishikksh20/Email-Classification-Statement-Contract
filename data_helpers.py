import numpy as np
import re
import itertools
from collections import Counter
from os import listdir
from os.path import isfile, join


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads Categorical data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positiveFiles = [positive_data_file + f for f in listdir(positive_data_file) if isfile(join(positive_data_file, f))]
    negativeFiles = [negative_data_file + f for f in listdir(negative_data_file) if isfile(join(negative_data_file, f))]
    # x_text = [clean_str(sent) for sent in x_text]

    # Load data from files
    positive_examples=[]
    negative_examples=[]
    # positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # Split by words

    for pf in positiveFiles:
        with open(pf, "r", encoding='utf8', errors='ignore') as f:
            line=f.read()
        positive_examples.append(line)
    for nf in negativeFiles:
        with open(nf, "r", encoding='utf8', errors='ignore') as f:
            line=f.read()
        negative_examples.append(line)

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_test_data(data_file):
    """
    Loads Test data from test folder, splits the data into words.
    Returns split sentences.
    """
    # Load data from files
    testFiles = [data_file + f for f in listdir(data_file) if isfile(join(data_file, f))]

    # Load data from files
    test_examples=[]

    for pf in testFiles:
        with open(pf, "r", encoding='utf8', errors='ignore') as f:
            line=f.read()
        test_examples.append(line)

    test_examples = [s.strip() for s in test_examples]
    x_text = [clean_str(sent) for sent in test_examples]

    return x_text

def load_test_file(test_file):
    """
    Loads Test file, splits the data into words.
    Returns split sentences.
    """

    # Load data from files
    test_examples=[]
    with open(test_file, "r", encoding='utf8', errors='ignore') as f:
        line=f.read()
    test_examples.append(line)

    test_examples = [s.strip() for s in test_examples]
    x_text = [clean_str(sent) for sent in test_examples]

    return x_text

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
