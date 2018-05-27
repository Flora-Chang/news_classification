#encoding: utf-8
import tensorflow as tf

class Model(object):
    def __init__(self,
                 batch_size,
                 embedding_size,
                 filter_size,
                 learning_rate,
                 headline_len_threshold,
                 has_text=False,
                 text_len_threshold=None,
                 word_vec_initializer=None):
        """
        initialize the attributes of the model
        :param batch_size: the num of training examples for one step 
        :param embedding_size: embedding dimension
        :param filter_size: the num of convolution filters
        :param learning_rate  
        :param headline_len_threshold: the max length of the headline
        :param has_text: whether has news text or only has headline
        :param text_len_threshold: the max length of the text
        :param word_vec_initializer: the pre_trained word embeddings, and it's constant during train time
        """
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.filter_size = filter_size
        self.lr = learning_rate
        self.headline_len_threshold = headline_len_threshold
        self.text_len_threshold = text_len_threshold
        self.word_vec_initializer = word_vec_initializer
        self.has_text=has_text
        self.formard()
        #self.predict()

    def input_layer(self):
        """
        define placeholder for the model
        :param has_article: if true, inputs include headline and text, if false, only headline provided
        """
        with tf.variable_scope("Input_layer"):
            self.headlines = tf.placeholder(dtype=tf.int32,
                                            shape=(None, self.headline_len_threshold),
                                            name='headline')
            self.headlines_len = tf.placeholder(dtype=tf.int32,
                                            shape=(None),
                                            name='headline_len')
            self.artificial_features = tf.placeholder(dtype=tf.float32,
                                            shape=(None, 78),
                                            name='artificial_feature')
            if self.has_text:
                self.texts = tf.placeholder(dtype=tf.int32,
                                           shape=(None, self.text_len_threshold),
                                           name='text')
                self.texts_len = tf.placeholder(dtype=tf.int32,
                                            shape=(None),
                                            name='text_len')
            self.labels = tf.placeholder(dtype=tf.int32,
                                        shape=(None),
                                        name='label')

    def embedding_layer(self, inputs, reuse=False):
        """
        according to the embedding matrix, translate the [batch_size, len_threshold] inputs to [batch_size, len_threshold, embedding_size]
        :param inputs: batch of input text, [batch_size, len_threshold]
        :return: a tensor after look up embedding matrix, [batch_size, len, embedding_size]
        """
        with tf.variable_scope('Embedding_layer'), tf.device("/cpu:0"):
            if reuse :
                tf.get_variable_scope().reuse_variables()
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                                    initializer=self.word_vec_initializer,
                                                    dtype=tf.float32,
                                                    trainable=False)
            #self.sm_emx_op = tf.summary.histogram('EmbeddingMatrix', self.embedding_matrix)
            print("inputs: ", inputs)
            embedding_str = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
            print("embedding_str: ", embedding_str)
            return embedding_str

    def feature_detection_cnn(self, inputs, len_threshold, filter_size, reuse=False):
        """
        the main module of this model, use CNN network to extract features from the text
        :param inputs: the input text after look up embedding matrix, [batch_size, len_threshold, embedding_size]
        :param len_threshold:  a scalar, the max length of the input
        :param filter_size: a scalar, the num of the Convolution kernels
        :param reuse: whether reuse the parameters or not
        :return: a tensor of [batch_size, filter_size/2], each row represents a distributed representation of the each input text
        """
        with tf.variable_scope("Feature_detection_layer"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            inputs = tf.reshape(inputs, [-1, len_threshold, self.embedding_size, 1])
            conv = tf.layers.conv2d(inputs=inputs,
                                     filters=filter_size,
                                     kernel_size=[3, self.embedding_size],
                                     activation=tf.nn.tanh,
                                     name="conv")  # [?, len_threshold-3+1, filter_size]
            pooling_size = len_threshold - 3 + 1
            pool = tf.layers.max_pooling2d(inputs=conv,
                                            pool_size=[pooling_size, 1],
                                            strides=[1, 1],
                                            name="pooling")  # [?, 1,1 filter_size]
            pool = tf.reshape(pool, [-1, filter_size])  # [?, filter_size]
            fc = tf.layers.dense(inputs=pool, units=filter_size/2, activation=tf.nn.tanh, name="fc_1")
            return fc

    def feature_detection_gru(self, inputs, inputs_len, num_units, reuse=False):
        """
        the main module of this model, use GRU and Self-Attention network to extract features from the text
        :param inputs: the input text after look up embedding matrix, [batch_size, len_threshold, embedding_size]
        :param inputs_len: the actual sequence length, [batch_size]
        :param num_units: the units of the RNN Cell
        :param reuse: whether reuse the parameters or not
        :return: a tensor of [batch_size, num_units*2], each row represents a distributed representation of the each input text
        """
        with tf.variable_scope("Encoding_layer") as scope1:
            if reuse:
                tf.get_variable_scope().reuse_variables()
                print("inputs_len: ", inputs_len)
            cell_fw_1 = tf.contrib.rnn.GRUCell(num_units=num_units)
            cell_bw_1 = tf.contrib.rnn.GRUCell(num_units=num_units)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw_1,
                                                                     cell_bw=cell_bw_1,
                                                                     inputs=inputs,
                                                                     sequence_length=inputs_len,
                                                                     scope=scope1,
                                                                     dtype=tf.float32)
            outputs = tf.concat(outputs, 2) # [batch_size, len_threshold, num_units*2]

        with tf.variable_scope("Self_matching_layer") as scope2:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            #self-match matrix, [batch_size, len_threshold, len_threshold]
            match_matrix = tf.matmul(outputs, outputs, transpose_b=True, name="match_matrix")
            inputs_mask = self.mask(inputs, inputs_len)  #[batch_size, len_threshold]
            #attention matrix, [batch_size, len_threshold, 2*num_units+embedding_size]
            attention_matrix = tf.concat([tf.matmul(tf.nn.softmax(match_matrix, axis=-1),
                                                outputs) , inputs], axis=-1) * tf.expand_dims(inputs_mask, 2)
            cell_fw_2 = tf.contrib.rnn.GRUCell(num_units=num_units)
            cell_bw_2 = tf.contrib.rnn.GRUCell(num_units=num_units)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw_2,
                                                                     cell_bw=cell_bw_2,
                                                                     inputs=attention_matrix,
                                                                     sequence_length=inputs_len,
                                                                     scope=scope2,
                                                                     dtype=tf.float32)
            output_states = tf.concat(output_states, 1) #[batch_size, num_units*2]
            return output_states

    def fully_connected(self, hidden_states, artificial_features, reuse=False):
        """
        :param hidden_states: CNN/RNN outputs, [batch_size, hidden_state_size]
        :param artificial_features: artificial features, [batch_size, 78]
        :return: fc_2, a tensor of [batch_size, 7], No softmax
        """
        with tf.variable_scope("Fully_connected_layer"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            feature = tf.concat([hidden_states, artificial_features], axis=-1)
            fc_1 = tf.layers.dense(inputs=feature,
                                 units=self.filter_size,
                                 activation=tf.nn.tanh,
                                 name="fc_1")
            fc_2 = tf.layers.dense(inputs=fc_1,
                                     units=7,
                                     activation=tf.nn.tanh,
                                     name="fc_2")
            return fc_2

    def mask(self, inputs, inputs_len):
        inputs_lens = tf.reshape(inputs_len, shape=[-1, 1])
        range = tf.expand_dims(tf.range(0, inputs.get_shape().as_list()[1]), 0)
        mask = tf.less(range, inputs_lens)
        result = tf.cast(mask, dtype=tf.float32)
        return result

    def formard(self, reuse=False):
        """the forward propagation process"""
        self.input_layer()

        self.embedding_headline = self.embedding_layer(self.headlines, reuse=reuse)
        print("embedding_headline: ", self.embedding_headline)
        print("headline_len: ", self.headlines_len)
        #self.headline_feature = self.feature_detection_cnn(self.embedding_headline, self.headline_len_threshold, self.filter_size, reuse=reuse)
        self.headline_feature = self.feature_detection_gru(self.embedding_headline, self.headlines_len, self.filter_size)
        self.feature = self.headline_feature

        if self.has_text:
            self.embedding_text = self.embedding_layer(self.texts, reuse=reuse)
            #self.text_feature = self.feature_detection_cnn(self.embedding_text, self.text_len_threshold, self.filter_size, reuse=reuse)
            self.text_feature = self.feature_detection_gru(self.embedding_text, self.headlines_len, self.filter_size, reuse=reuse)
            self.feature = tf.concat([self.feature, self.text_feature], axis=-1)

        self.output = self.fully_connected(self.feature, self.artificial_features, reuse=reuse)

        self.predictions = tf.cast(tf.argmax(self.output, 1), dtype=tf.int32, name="prediction")
        print("predictions: ", self.predictions)

        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels)
        self.loss = tf.reduce_mean(self.losses, name="losses")

        self.optimize_op = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                  beta1=0.90,
                                                  beta2=0.999,
                                                  epsilon=1e-08).minimize(self.loss)

    def predict(self, reuse=True):
        """the forward propagation process"""

        embedding_headline = self.embedding_layer(self.headlines, reuse=reuse)
        # self.headline_feature = self.feature_detection_cnn(self.embedding_headline, self.headline_len_threshold, self.filter_size, reuse=reuse)
        headline_feature = self.feature_detection_gru(embedding_headline, self.headlines_len,
                                                           self.filter_size, reuse=reuse)
        feature = headline_feature

        if self.has_text:
            embedding_text = self.embedding_layer(self.texts, reuse=reuse)
            # self.text_feature = self.feature_detection_cnn(self.embedding_text, self.text_len_threshold, self.filter_size, reuse=reuse)
            text_feature = self.feature_detection_gru(embedding_text, self.headlines_len, self.filter_size,
                                                           reuse=reuse)
            feature = tf.concat([feature, text_feature], axis=-1)

        output = self.fully_connected(feature, self.artificial_features, reuse=reuse)

        predictions = tf.cast(tf.argmax(output, 1), dtype=tf.int32, name="prediction_test")
        print("predictions: ", predictions)





