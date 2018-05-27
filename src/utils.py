#coding: utf-8
import tensorflow as tf

flags = tf.app.flags

#change these parameters at each running
flags.DEFINE_string("flag", "version_1", "version flag")
flags.DEFINE_string("model_name", "version_1_lr_0.002_bz_64_filter_64_embedding_100_1527413274", "used to look for the model path")


# Model parameters
flags.DEFINE_integer("filter_size", 64, "the num of filters of CNN")
flags.DEFINE_integer("embedding_size", 100, "word embedding size")
flags.DEFINE_boolean("has_text", False, "if True, input contains text and the headline, if False, only headline")

#Train/ Test parameters
flags.DEFINE_integer("headline_len_threshold", 30, "max length of the headline")
flags.DEFINE_integer("text_len_threshold", 200, "max length of the text")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("epochs", 10, "number of training epochs")
flags.DEFINE_integer("validation_steps", 100, "steps between validations")
flags.DEFINE_float("learning_rate", 0.002, "learning rate")

#directories
flags.DEFINE_string("train_set", "../data/with_artificial_data/train.shuf.txt", "training set path")
flags.DEFINE_string("train_test_set", "../data/with_artificial_data/train.test.txt", "sample 500 examples from the training set")
flags.DEFINE_string("dev_set", "../data/with_artificial_data/dev.txt", "dev set path")
flags.DEFINE_string("test_set", "../data/with_artificial_data/test.txt", "test set path")
flags.DEFINE_string("vocab_path", "../data/vocab.txt", "vocabulary path")
flags.DEFINE_string("vectors_path", "../data/vector.txt", "word embedding path")
flags.DEFINE_string("save_dir", "../logs/", "save directory")
flags.DEFINE_string("predict_dir", "../predict/my_predict.csv", "result directory")
flags.DEFINE_string("load_steps", "model.ckpt-300.meta", "model file name")



FLAGS = flags.FLAGS