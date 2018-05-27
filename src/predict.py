# encoding: utf-8
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from model import Model
from loaddata import get_word_dict, get_label_dict, LoadTrainData, LoadTestData, get_word_vector
from test import test
from utils import FLAGS
import sys

# load the vocabulary and word vector
word_dict = get_word_dict()
word_vec = get_word_vector()

with tf.Session() as sess:
    #change model name when you run predict
    model_path = os.path.join(FLAGS.save_dir, FLAGS.model_name)
    # loading model structure, parameters and variables
    print("importing model...")
    saver = tf.train.import_meta_graph(os.path.join(model_path, FLAGS.load_steps))

    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    # print the variable names 
    all_vars = tf.trainable_variables()
    #for v in all_vars:
    #    print(v.name)
    for op in sess.graph.get_operations():
        if "prediction" in op.name:
            print(op.name)

    headlines = graph.get_tensor_by_name("Input_layer/headline:0")
    headlines_len = graph.get_tensor_by_name("Input_layer/headline_len:0")
    artificial_features = graph.get_tensor_by_name("Input_layer/artificial_feature:0")
    if FLAGS.has_text:
        texts = graph.get_tensor_by_name("Input_layer/text:0")
        texts_len = graph.get_tensor_by_name("Input_layer/text_len:0")
    labels = graph.get_tensor_by_name("Input_layer/label:0")
    predictions = graph.get_tensor_by_name("prediction:0")

    testing_set = LoadTestData(word_dict,
                           data_path=FLAGS.test_set,
                           label_dict=get_label_dict(),
                           headline_len_threshold=FLAGS.headline_len_threshold,
                           text_len_threshold=FLAGS.text_len_threshold,
                           batch_size=FLAGS.batch_size)
    total_data = 0
    right = 0
    cnt = 0
    for batch_data in testing_set.next_batch():
        feed_dict = {headlines: batch_data[1],
                     labels: batch_data[0],
                     headlines_len: batch_data[2],
                     artificial_features: batch_data[3]}

        if FLAGS.has_text:
            feed_dict[texts] = batch_data[4]
            feed_dict[texts_len] = batch_data[5]
        res = sess.run(predictions, feed_dict)
        df = pd.DataFrame(res, columns=['prediction'])
        total_data += len(df)
        if cnt ==0:
            df.to_csv(FLAGS.predict_dir, mode='a', index=False)
            cnt += 1
        else:
            df.to_csv(FLAGS.predict_dir, mode='a', index=False, header=False)

print("total data: ", total_data)
