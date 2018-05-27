#encoding: utf-8
import tensorflow as tf
import pandas as pd
import numpy as np

def test(sess, model, testing_set, filename=None):
    count = 0   # the num of examples in the testing set
    right = 0   # the num of correct predictions in the testing set
    flag=0
    right_list = [0 for i in range(0, 7)]   # the num of correct predictions for each class in the testing set
    count_list = [0 for i in range(0, 7)]   # the num of examples for each class in the testing set

    for batch_data in testing_set.next_batch(shuffle=False):
        labels = batch_data[0]
        feed_dict = {model.labels: batch_data[0],
                     model.headlines: batch_data[1],
                     model.headlines_len: batch_data[2],
                     model.artificial_features: batch_data[3]}
        if len(batch_data) == 6:
            feed_dict[model.texts] = batch_data[4]
            feed_dict[model.texts_len] = batch_data[5]
        res = sess.run([model.predictions], feed_dict)
        res = list(zip(labels, res[0].tolist()))
        df = pd.DataFrame(res, columns=['label', 'predict'])

        for i in range(0, len(df)):
            label = df['label'][i]
            predict = df['predict'][i]
            if label == predict:
                right += 1
                right_list[label] += 1
            count += 1
            count_list[label] += 1

        if filename is not None:
            if flag ==0:
                df.to_csv(filename, mode='a', index=False)
                flag += 1
            else:
                df.to_csv(filename, mode='a', index=False, header=False)

    print("accuracy：", right / (count + 0.001))
    for i, r, c in zip(range(0, 7), right_list, count_list):
        print("class : ", i)
        print("acc : ", r/(c + 0.001))

    print("=" * 60)
    return right / (count + 0.001)


