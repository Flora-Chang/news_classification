#encoding: utf-8
import numpy as np
from utils import FLAGS
import sys
sys.path.append("..")
from preprocess.artifical_featurre import key_words, create_artificial_feature
import jieba
jieba.load_userdict("../data/names.txt")

def get_word_dict(input_file=FLAGS.vocab_path):
    """
    translate the vocabulary file into word_dict
    :param input_file: a file contains the vocabulary, one word in a line
    :return: word_dict,  word is the key, value is in range(0, len(vocab))
    """
    if not input_file:
        print("please enter the vocabulary path!")
        return None
    word_dict = {}
    cnt = 0
    with open(input_file) as f:
        for word in f:
            word_dict[word.strip()] = cnt
            cnt += 1
    return word_dict

def get_word_vector(input_file=FLAGS.vectors_path):
    """
    store the word embedding into an array
    :param input_file: file contains the word embeddings, a word vector in a line and split with " "
    :return: the embedding array
    """
    if not input_file:
        print("please enter the embeddings path!")
        return None
    word_vec = []
    with open(input_file) as f:
        for line in f:
            line = [float(v) for v in line.strip().split()]
            word_vec.append(line)
    return word_vec

def get_label_dict():
    label_dict = {"资讯热点": 0,
                  "电影电视剧评论": 1,
                  "人物深扒专访": 2,
                  "组图盘点": 3,
                  "明星街拍": 4,
                  "行业报道或分析": 5,
                  "机场图": 6}
    return label_dict

class LoadData(object):
    """class for data load"""
    def __init__(self, word_dict, data_path, label_dict,
                 headline_len_threshold=20,
                 text_len_threshold=200,
                 batch_size=64):
        self.word_dict = word_dict
        self.batch_size = batch_size
        self.headline_len_threshold = headline_len_threshold
        self.text_len_threshold = text_len_threshold
        self.data = open(data_path, "r").readlines()
        self.batch_size = batch_size
        self.label_dict = label_dict
        #self.batch_index = 0

    def word_2_id(self, word):
        """ return the word to index"""
        if word in self.word_dict:
            res = self.word_dict[word]
        else:
            res = self.word_dict['</s>']
        return res

    def padding(self, inputs, len_threshold):
        """
        padding the sequence into the same length of len_threshold with 0
        :param inputs: a batch sequence with different length
        :param len_threshold:  allowed max length of the sequence
        :return: 
            inputs_batch_major: a batch sequence with the same length, use numpy array
            seq_len: the actual sequence length
        """
        inputs_batch_major = np.zeros(shape=[len(inputs), len_threshold], dtype=np.int32)
        seq_len = np.array([len(seq) for seq in inputs])
        for i, seq in enumerate(inputs):
            for j, word in enumerate(seq):
                if j >= len_threshold:
                    seq_len[i] = len_threshold
                    break
                inputs_batch_major[i][j] = word
        return inputs_batch_major, seq_len



class LoadTrainData(LoadData):
    """
    class for train data load
    file format: 'label \t headline \t text \n', the 'text' can not exist
    """
    def next_batch(self, shuffle=True):
        self.batch_index = 0
        self.cnt = 0
        data = np.array(self.data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1
        if shuffle:
            np.random.shuffle(self.data)
        print("data_size: ", data_size)
        print("num_batches_per_epoch: ", num_batches_per_epoch)
        print("first data: ", self.data[0])
        while (self.batch_index < num_batches_per_epoch
               and (self.batch_index + 1) * self.batch_size <= data_size):
            headlines = []
            texts = []
            labels = []
            artificial_features = []
            start_index = self.batch_index * self.batch_size
            self.batch_index += 1
            end_index = min(self.batch_index * self.batch_size, data_size)
            batch_data = data[start_index : end_index]

            for line in list(batch_data):
                line = line.strip().split('\t')
                self.cnt += 1
                if len(line) < 1:
                    continue
                labels.append(self.label_dict[line[0]])
                headline = list(map(self.word_2_id, line[1].strip().split()))
                headlines.append(headline)
                artificial_feature = list([int(i) for i in line[2].strip().split(" ")])
                artificial_features.append(artificial_feature)
                if(len(line)) > 3:
                    text = list(map(self.word_2_id, line[2].strip().split()))
                    texts.append(text)
            headlines, headlines_len = self.padding(headlines, self.headline_len_threshold)
            artificial_features = np.array(artificial_features)
            if texts:
                texts, texts_len = self.padding(texts, self.text_len_threshold)
                yield labels, headlines, headlines_len, artificial_features, texts, texts_len
            else:
                yield labels, headlines, headlines_len, artificial_features


class LoadTestData(LoadData):
    """
    class for test data load, the data need not to be shuffled.
    file format: 'headline \t text \n', the 'text' can not exist'
    """
    def next_batch(self, shuffle=False):
        self.batch_index = 0
        self.cnt = 0
        data = np.array(self.data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1
        if shuffle:
            np.random.shuflle(self.data)

        while (self.batch_index < num_batches_per_epoch
               and (self.batch_index) * self.batch_size < data_size):
            headlines = []
            texts = []
            labels = []
            artificial_features = []
            start_index = self.batch_index * self.batch_size
            self.batch_index += 1
            end_index = min(self.batch_index * self.batch_size, data_size)
            batch_data = data[start_index: end_index]

            for line in list(batch_data):
                line = line.strip().split('\t')
                artificial_feature = create_artificial_feature(line[0])
                artificial_features.append(artificial_feature)
                headline = jieba.cut(line[0], cut_all=False)
                headline = list(map(self.word_2_id, headline))
                labels.append(0) #just a placeholder, no meaning
                headlines.append(headline)
                if (len(line)) >= 2:
                    text = jieba.cut(line[1], cut_all=False)
                    text = list(map(self.word_2_id, text))
                    texts.append(text)
            headlines, headlines_len = self.padding(headlines, self.headline_len_threshold)
            artificial_features = np.array(artificial_features)
            # note: label is all zeros, has no meaning
            print("headlines, headlines_len: ", np.shape(headlines), np.shape(headlines_len))
            if texts:
                texts, texts_len = self.padding(texts, self.text_len_threshold)
                yield labels, headlines, headlines,  artificial_features, texts, texts_len
            else:
                yield labels, headlines, headlines_len, artificial_features



