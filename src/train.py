#encoding: utf-8
import os
import time
import tensorflow as tf
from utils import FLAGS
from model import Model
from loaddata import get_word_dict, get_word_vector, get_label_dict, LoadTrainData
from test import test

word_dict = get_word_dict()
word_vec = get_word_vector()

training_set = LoadTrainData(word_dict,
                             data_path=FLAGS.train_set,
                             label_dict=get_label_dict(),
                             headline_len_threshold=FLAGS.headline_len_threshold,
                             text_len_threshold=FLAGS.text_len_threshold,
                             batch_size=FLAGS.batch_size
                             )

with tf.Session() as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ", timestamp)
    model_name = "{}_lr_{}_bz_{}_filter_{}_embedding_{}_{}".format(FLAGS.flag,
                                                                   FLAGS.learning_rate,
                                                                   FLAGS.batch_size,
                                                                   FLAGS.filter_size,
                                                                   FLAGS.embedding_size,
                                                                   timestamp)
    log_dir = "../logs/" + model_name
    model_path = os.path.join(log_dir, "model.ckpt")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model = Model(batch_size=FLAGS.batch_size,
                  embedding_size=FLAGS.embedding_size,
                  filter_size=FLAGS.filter_size,
                  learning_rate=FLAGS.learning_rate,
                  headline_len_threshold=FLAGS.headline_len_threshold,
                  has_text=FLAGS.has_text,
                  text_len_threshold=None,
                  word_vec_initializer=word_vec)


    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    saver = tf.train.Saver(tf.global_variables())
    before_acc = 0.0
    for epoch in range(FLAGS.epochs):
        print("epoch: ", epoch)
        for batch_data in training_set.next_batch(shuffle=True):
            feed_dict = {model.labels: batch_data[0],
                         model.headlines: batch_data[1],
                         model.headlines_len: batch_data[2],
                         model.artificial_features: batch_data[3]}
            if len(batch_data) == 6:
                feed_dict[model.texts] =  batch_data[4]
                feed_dict[model.texts_len] = batch_data[5]
            _, loss, predictions = sess.run([model.optimize_op, model.loss, model.predictions],
                                            feed_dict)
            if step % FLAGS.validation_steps == 0:
                print("step: ", step)
                print("loss: ", loss)
                train_test_set = LoadTrainData(word_dict,
                                       data_path=FLAGS.train_test_set,
                                       label_dict=get_label_dict(),
                                       headline_len_threshold=FLAGS.headline_len_threshold,
                                       text_len_threshold=FLAGS.text_len_threshold,
                                       batch_size=FLAGS.batch_size)
                # test on the training set to compare with the acc on the dev set to prevent overfit
                print("on training set: ")
                acc_train = test(sess, model, train_test_set)

                dev_set = LoadTrainData(word_dict,
                                       data_path=FLAGS.dev_set,
                                       label_dict=get_label_dict(),
                                       headline_len_threshold=FLAGS.headline_len_threshold,
                                       text_len_threshold=FLAGS.text_len_threshold,
                                       batch_size=FLAGS.batch_size)

                #test on the dev set to implement early-stopping
                print("on validation set: ")
                acc_dev = test(sess, model, dev_set)

                #save best model
                if acc_dev > before_acc:
                    before_acc = acc_dev
                    saver_path = saver.save(sess, model_path, step)
            step += 1
    coord.request_stop()
    coord.join(threads)






