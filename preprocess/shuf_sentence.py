#encoding: utf-8
import sys
import random

def shuf_sent(sentence):
    """randomly transform the order of clauses in sentences to increase diversity"""
    sentence = sentence.split(" ")
    if len(sentence) == 1:
        sentence = sentence[0].split("，")
        if len(sentence) == 1:
            return sentence[0]
    ori_sentence = sentence[:]
    while sentence == ori_sentence:
        random.shuffle(sentence)
    return  " ".join(sentence)

if __name__ == "__main__":
    src_file = "../data/last_col.txt"
    dst_file = "../data/shuffle_sen.txt"
    with open(src_file, "r") as src_f, open(dst_file, "w") as dst_f:
        for line in src_f:
            sentence = shuf_sent(line.strip())
            dst_f.write(sentence+"\n")