#encoding: utf-8
import sys
import random

index = [i for i in range(0, 9980)]
random.shuffle(index)
index = index[:500]

def select_500(src_file, train_file, dev_file):
    with open(src_file) as src_f, open(train_file, 'w') as train_f, open(dev_file, 'w') as dev_f:
        for i, line in enumerate(src_f.readlines()):
            if i in index:
                dev_f.write(line.strip() + "\n")
            else:
                train_f.write(line.strip() + "\n")

if __name__ == '__main__':
    src_shuf_file = '../data/shuf_trans_ori_data/shuffle_sen.txt'
    train_shuf_file = '../data/shuf_trans_ori_data/shuffle_train.txt'
    dev_shuf_file = '../data/shuf_trans_ori_data/shuffle_dev.txt'
    src_trans_file = '../data/shuf_trans_ori_data/trans_corpus.txt'
    train_trans_file = '../data/shuf_trans_ori_data/trans_train.txt'
    dev_trans_file = '../data/shuf_trans_ori_data/trans_dev.txt'
    src_ori_file = '../data/shuf_trans_ori_data/last_col.txt'
    train_ori_file = '../data/shuf_trans_ori_data/ori_train.txt'
    dev_ori_file = '../data/shuf_trans_ori_data/ori_dev.txt'
    src_label = "../data/shuf_trans_ori_data/label.txt"
    train_label = "../data/shuf_trans_ori_data/label_train.txt"
    dev_label = "../data/shuf_trans_ori_data/label_dev.txt"

    select_500(src_shuf_file, train_shuf_file, dev_shuf_file)
    select_500(src_trans_file, train_trans_file, dev_trans_file)
    select_500(src_ori_file, train_ori_file, dev_ori_file)
    select_500(src_label, train_label, dev_label)





