from collections import defaultdict
from six import iteritems
import numpy as np
import json
import tensorflow as tf
import os
import time
from tensorflow import keras


class PopularModel(object):
    def __init__(self, batch_size, item_size, read_size, follow_size):
        self._batch_size = batch_size
        self._item_size = item_size
        self._item_dimension = 100

        self._read_size = read_size
        self._read_dimension = 50

        self._follow_size = follow_size
        self._follow_dimension = 50

        self.sess = None
        self.saver = None
        self.graph = tf.Graph()

        with tf.device("/device:GPU:0"):
            with self.graph.as_default():
                with tf.variable_scope('inputs'):
                    self.inputs = tf.placeholder(
                        dtype=tf.int32,
                        shape=[self._batch_size, None],
                        name='inputs'
                    )
                    self.read_inputs = tf.placeholder(
                        dtype=tf.int32,
                        shape=[self._batch_size, None],
                        name='read_inputs'
                    )
                    self.labels = tf.placeholder(
                        dtype=tf.float32,
                        shape=[self._batch_size, None],
                        name='labels'
                    )
                    self.follows = tf.placeholder(
                        dtype=tf.int32,
                        shape=[self._batch_size, None],
                        name='inputs'
                    )

                with tf.variable_scope('inputs_embedding'):
                    inputs_embeddings = tf.Variable(
                        tf.truncated_normal(shape=[self._item_size, self._item_dimension],
                                            mean=-1.0,
                                            stddev=1.0,
                                            dtype=tf.float32)
                    )
                    inputs_embedding = tf.nn.embedding_lookup(inputs_embeddings, self.inputs)
                    inputs_component = tf.reduce_sum(inputs_embedding, axis=1)

                with tf.variable_scope('reads_embedding'):
                    reads_embeddings = tf.Variable(
                        tf.truncated_normal(shape=[self._read_size, self._read_dimension],
                                            mean=-1.0,
                                            stddev=1.0,
                                            dtype=tf.float32)
                    )

                    read_embedding = tf.nn.embedding_lookup(reads_embeddings, self.read_inputs)
                    read_component = tf.reduce_sum(read_embedding, axis=1)

                with tf.variable_scope('follow_embedding'):
                    follows_embeddings = tf.Variable(
                        tf.truncated_normal(shape=[self._follow_size, self._follow_dimension],
                                            mean=-1.0,
                                            stddev=1.0,
                                            dtype=tf.float32)
                    )

                    follow_embedding = tf.nn.embedding_lookup(follows_embeddings, self.follows)
                    follow_component = tf.reduce_sum(follow_embedding, axis=1)

                with tf.variable_scope('forward'):
                    self.marge = tf.concat([inputs_component, read_component, follow_component], axis=1)
                    self.dense_output = tf.layers.dense(self.marge, 200)
                    self.dense_output += tf.random_normal(shape=tf.shape(self.dense_output), dtype=tf.float32)
                    self.dense_output = tf.layers.dense(self.dense_output, 128, tf.nn.leaky_relu)
                    self.dense_output = tf.layers.dense(self.dense_output, 64, activity_regularizer=keras.regularizers.l2(l=0.1))
                    self.dense_output = tf.layers.dense(self.dense_output, 64, tf.nn.leaky_relu)
                    self.dense_output = tf.layers.dense(self.dense_output, 128, tf.nn.leaky_relu)
                    self.dense_output = tf.layers.dense(self.dense_output, self._item_size)
                    self.output = tf.sigmoid(self.dense_output)

                with tf.variable_scope('optimizer'):
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output)
                    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.dense_output)
                    self.train = tf.train.AdamOptimizer(learning_rate=1e-7).minimize(loss)
                    self.loss = tf.reduce_mean(loss)

                init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.graph)
            self.sess.run(init)

    def predict(self, inputs=None, read_inputs=None, follows=None):
        output = self.sess.run(
            self.output,
            feed_dict={
                self.inputs: inputs,
                self.read_inputs: read_inputs,
                self.follows: follows
            }
        )
        return output[0]

    def model_train(self, inputs, read_inputs, labels, follows):
        self.sess.run(
            self.train,
            feed_dict={
                self.inputs: inputs,
                self.read_inputs: read_inputs,
                self.labels: labels,
                self.follows: follows
            }
        )

    def close(self):
        self.sess.close()

    def model_save(self, path):
        path = '/home/wooktae/data/dl_model/kakao/model2.ckpt'
        self.saver.save(self.sess, path)

    def model_load(self, path):
        path = '/home/wooktae/data/dl_model/kakao/model2.ckpt'
        self.saver.restore(self.sess, path)


if __name__ == "__main__":
    from tqdm import tqdm

    np.seterr(divide='ignore', invalid='ignore')

    batch_size = 50
    train_data, test_data, all_data = get_data()
    meta_data, read_data, meta_read_data = get_metadata()
    follow_data, follow_len = get_follow()

    meta_data_len = len(meta_data)
    read_data_len = len(read_data)
    model = PopularModel(batch_size, meta_data_len, read_data_len, follow_len)


    def max_batch_func(batch_data: list):
        max_len = 1
        for batch_datum in batch_data:
            if max_len < len(batch_datum):
                max_len = len(batch_datum)

        batch = []
        for batch_datum in batch_data:
            batch_pad = np.array(np.pad(np.array(batch_datum), (0, max_len - len(batch_datum)), mode='constant') == 0) * meta_data_len
            batch_value = np.pad(np.array(batch_datum), (0, max_len - len(batch_datum)), mode='constant')
            batch_value += batch_pad
            batch.append(batch_value)

        return batch


    def train_func():
        cnt = 0
        sum_predict = []
        train_value_batch = []
        train_keyword_batch = []
        label_value_batch = []
        follow_value_batch = []
        for k, v in tqdm(train_data.items()):
            if not (v.get('train') and v.get('label')):
                continue

            train_value_list = []
            train_keyword_list = []
            for train_value in v.get('train'):
                if train_value in meta_data:
                    train_value_idx = meta_read_data[train_value].get('key')
                    train_value_list.append(train_value_idx)
                    train_keyword_list += meta_read_data[train_value].get('read')
            train_value_list = list(set(train_value_list))

            if not train_value_list:
                train_value_list.append(meta_data_len)
            if not train_keyword_list:
                train_keyword_list.append(meta_data_len)

            label_value_list = []
            reg_value_list = []
            for label_value in v.get('label'):
                if label_value in meta_data:
                    label_value_idx = meta_read_data[label_value].get('key')
                    reg_value = meta_read_data[label_value].get('reg')
                    label_value_list.append(label_value_idx)
                    reg_value_list.append(reg_value)
            if not reg_value_list:
                reg_value_list.append(0)

            follow_value_list = []
            if k in follow_data:
                follow_value_list += follow_data[k]
            else:
                follow_value_list.append(follow_len)

            reg_mean = np.mean(reg_value_list, dtype=np.float16)
            labels_mean = np.array(np.bincount(label_value_list, minlength=meta_data_len) >= 1,
                              dtype=np.float64) * 0.9 + 0.05
            labels_mean = labels_mean * reg_mean

            label_value_batch.append(labels_mean)
            train_value_batch.append(train_value_list)
            train_keyword_batch.append(train_keyword_list)
            follow_value_batch.append(follow_value_list)

            if len(train_value_batch) == batch_size:
                inputs = max_batch_func(train_value_batch)
                labels = label_value_batch
                keywords = max_batch_func(train_keyword_batch)
                follows = max_batch_func(follow_value_batch)
                model.model_train(inputs, keywords, labels, follows)

                if cnt % 10 == 0:
                    result = model.predict(inputs, keywords, follows)
                    sum_predict.append((np.sum(labels * result) / np.sum(labels)))

                if cnt % 100 == 0:
                    sum_predict_result = np.mean(sum_predict)
                    print(sum_predict_result)
                    sum_predict = []

                label_value_batch = []
                train_value_batch = []
                train_keyword_batch = []
                follow_value_batch = []
                cnt += 1


    for _ in range(1):
        train_func()
    model.model_save('')
    model.close()
    model = PopularModel(1, meta_data_len, read_data_len)

    model.model_load('')
    meta_key = np.array(list(meta_data.keys()))
    with open(r'../dev.users') as f:
        recom_dev_data = f.read()
        recom_dev_data = recom_dev_data.split('\n')
        recom_dev_data = recom_dev_data[:3000]

    with open(r'../nf.recommender.txt', 'w') as f:
        for user in tqdm(recom_dev_data):
            user = user.replace('\n', '')
            try:
                k = user
                if k not in all_data:
                    raise ImportError
                v = all_data[user]

                test_value_list = []
                test_keyword_list = []
                for test_value in v:
                    if test_value in meta_read_data:
                        test_value_idx = meta_read_data[test_value].get('key')
                        test_value_list.append(test_value_idx)
                        test_keyword_list += meta_read_data[test_value].get('read')

                follow_value_list = []
                if k in follow_data:
                    follow_value_list += follow_data[k]
                else:
                    follow_value_list.append(follow_len)

                predict_result = model.predict([test_value_list], [test_keyword_list], [follow_value_list])
                sort_idx = np.argsort(predict_result)[::-1][:400]
                score_result = meta_key[sort_idx]

                result = [str(x) for x in score_result if str(x) not in ' '.join(v)][:100]

                result.insert(0, k)
                result_string = ' '.join(result)
                f.write(result_string + '\n')
            except Exception as e:
                print('null: ', k)
                result_string = " @brunch_151 @sweetannie_145 @chofang1_15 @seochogirl_16 @seochogirl_1 @seochogirl_18 @seochogirl_17 @conbus_43 @seochogirl_11 @hjl0520_26 @seochogirl_12 @seochogirl_13 @seochogirl_14 @seochogirl_15 @wootaiyoung_85 @seochogirl_10 @intlovesong_28 @steven_179 @tenbody_1164 @shindong_38 @tenbody_1305 @seochogirl_8 @seochogirl_7 @shanghaiesther_46 @seochogirl_6 @noey_130 @seochogirl_29 @seochogirl_9 @bzup_281 @seochogirl_2 @seochogirl_3 @hongmilmil_33 @roysday_314 @seochogirl_4 @seochogirl_5 @deckey1985_51 @ohmygod_42 @hyehyodam_19 @boot0715_115 @fuggyee_108 @dailylife_207 @mightysense_9 @syshine7_57 @wikitree_54 @sweetannie_146 @seochogirl_20 @hjl0520_28 @roysday_307 @roysday_313 @13july_92 @brunch_149 @dailylife_219 @aemae-human_15 @ladybob_30 @tamarorim_133 @anetmom_52 @keeuyo_57 @sunnysohn_60 @kidjaydiary_6 @moment-yet_155 @yoriyuri_12 @thebluenile86_4 @ladybob_29 @13july_94 @jijuyeo_9 @namgizaa_46 @anti-essay_150 @thinkaboutlove_234 @dailylife_178 @aemae-human_9 @anetmom_47 @honeytip_945 @hjl0520_27 @moment-yet_157 @dreamwork9_25 @keeuyo_56 @dryjshin_255 @mariandbook_413 @syshine7_56 @boot0715_111 @scienceoflove_5 @seochogirl_28 @everysoso_8 @drunk-traveler_49 @taekangk_61 @ggpodori_14 @dancingsnail_64 @dancingsnail_65 @mentorgrace_8 @juck007_32 @kotatsudiary_66 @studiocroissant_43 @moondol_222 @hum0502_9 @tamarorim_131 @honeytip_841 @dailylife_666 @mothertive_72 @winebee_87 @wonderland_131"
                result_string = str(k) + result_string
                f.write(result_string + '\n')
