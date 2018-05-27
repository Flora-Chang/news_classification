#encoding: utf-8
import sys

key_words = ["快讯", "影视", "电影", "电视", "综艺", "节目", "音乐",
                       "开播", "播出", "热播", "收官", "上映", "新片",
                       "预告片", "MV", "单曲", "新歌","《", "》", "深扒",
                       "专访", "采访", "群访", "受访", "提问", "问题", "回答",
                       "媒体名", "盘点", "旧闻", "梳理", "情史", "回顾", "曾",
                       "昔日", "过去", "历史", "起底", "盘点", "写真", "街拍",
                       "封面", "大片", "时尚","潮流", "美照","混搭", "范儿", "型男",
                       "酷", "帅", "淑女","熟女", "霸气", "魅力", "清纯", "优雅",
                       "小清新", "性感", "曼妙", "魅惑", "沉稳", "肌肉", "尽显",
                       "演绎", "惊艳","吸睛", "前卫", "简约", "复古", "黑白",
                       "行业", "产业","影视公司","大佬", "机场", "现身", "行李"]

def create_artificial_feature(src):
    """
    keyword matching, if word in the key_words list is a sub_str of a line, 
    the corresponding position of the feature vector is set to 1 
    """
    artificial_feature = [0 for i in key_words]
    for i, word in enumerate(key_words):
        if word in src:
            artificial_feature[i] = 1
    return artificial_feature


if __name__ == "__main__":
    corpus = "../data/shuf_trans_ori_data/ori_train.txt"
    artificial_feature_file = "../data/shuf_trans_ori_data/ori_train.feature.txt"
    with open(corpus) as in_f, open(artificial_feature_file, 'w') as out_f:
        for line in in_f:
            artificial_feature = create_artificial_feature(line.strip())
            out_f.write(" ".join([str(i) for i in artificial_feature]) + "\n")
