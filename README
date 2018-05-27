A Tensorflow implementation of news multi-classification.

Dataset:
    (1). 9980 labeled news title, a total of 7 labels are included.
    (2). pre-trained word embeddings, using word2vec, skip-gram, embedding dimension is 100

Requirements:
    Python>=3.5
    TensorFlow>=1.5
    Numpy
    Pandas
    jieba
    urllib
    hashlib
    xlrd

Usage:

    (1) To train the model with this code, run:

    python3 train.py --flag "version_1" --batch_size 64 --headline_len_threshold 30 \
    --filter_size 64 --train_set '../data/train.txt' --dev_set '../data/dev.txt' \
    --vocab_path '../data/vocab.txt' --vectors_path '../data/vector.txt' --save_dir '../logs/'

    Note file format:
    train.txt & dev.txt: label \t news_headline \t news_text \n, the news_text can not exit
    vocab.txt: one word in a line
    vector.txt: one string in a line,  the elements split with " "

    (2) To use a pre-trained model to do prediction, run:

     python3 predict.py --flag "version_1" --batch_size 64 --headline_len_threshold 30 \
    --filter_size 64 --test_set '../data/test.txt' --vocab_path '../data/vocab.txt' \
    --vectors_path '../data/vector.txt' --save_dir '../logs/' --load_steps 'model.ckpt-5.meta'\
    --predict_dir '../predict/version_1.csv'

    Note file format:
    test.txt: news_headline \t news_text \n, the news_text can not exit

Tricks:
    1. use translate and shuffle sentence to extend corpus
    2. construct an artificial features just using keyword matching(details can be found in preprocess/artificial_feature.py)


Results:
    on 500 dev set, accuracy is 76.5%
    for each classification, the result is :
    label   label_dec           accuracy        precision       recall
    0       资讯热点                80.9%
    1       电影电视剧评论           85.9%
    2       人物深扒专访             60%
    3       组图盘点                48.1%
    4       明星街拍                90.5%
    5       行业报道或分析           65.9%
    6       机场图                  92.8%

ToDo:
    (1). use some extra features like the num of pictures, the news_text
    (2). use a big corpus to train the neural network