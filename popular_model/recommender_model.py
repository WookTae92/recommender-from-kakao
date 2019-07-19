import tensorflow as tf
import numpy as np
import os
import pickle

from popular_model.batch_process import EmbeddingVocabulary, MetaDataBatch, TrainBatch, DevBatch
from popular_model.tensor_util import dynamic_rnn, dense_encoding, lstmcell


class PopularModel(object):
    def __init__(self,
                 batch_size: int,
                 predict_input_size: int,
                 predict_title_max_len: int,
                 vocabulary_size: int):

        self._batch_size = batch_size
        self._predict_input_size = predict_input_size
        self._predict_title_max_len = predict_title_max_len
        self._vocabulary_size = vocabulary_size
        self._embedding_dimension = 300

        self._masking_matrix = np.array([x == y for x in range(self._batch_size) for y in range(self._batch_size)])
        self._masking_matrix = np.reshape(self._masking_matrix, [self._batch_size, self._batch_size])

        self._model_input = dict()
        self._model_output = dict()

        self.sess = None
        self.saver = None
        self.graph = tf.Graph()
        self.get_gpu_model()

    def get_gpu_model(self):
        with tf.device("/device:GPU:0"):
            with self.graph.as_default():
                with tf.variable_scope('inputs'):
                    masking = tf.constant(self._masking_matrix, dtype=tf.bool)
                    self._model_input['train'] = tf.placeholder(
                        dtype=tf.int32,
                        shape=[self._batch_size, None, None],
                        name='inputs'
                    )
                    self._model_input['target'] = tf.placeholder(
                        dtype=tf.int32,
                        shape=[self._batch_size, None],
                        name='targets'
                    )
                    self._model_input['positive'] = tf.placeholder(
                        dtype=tf.float64,
                        shape=[self._batch_size],
                        name='positive'
                    )
                    self._model_input['predict'] = tf.placeholder(
                        dtype=tf.int32,
                        shape=[self._predict_input_size, None],
                        name='predict_targets'
                    )
                    self._model_input['saved_title_embedding'] = tf.placeholder(
                        dtype=tf.float64,
                        shape=[self._batch_size, self._predict_input_size, 300],
                        name='saved_title_embedding'
                    )
                    self._model_input['dense_e'] = tf.placeholder(
                        dtype=tf.float64,
                        shape=[self._batch_size, 300],
                        name='saved_title_embedding'
                    )

                with tf.variable_scope('lstm_cell'):
                    cell, state = lstmcell(300, self._batch_size)

                with tf.variable_scope('embedding'):
                    embeddings = tf.Variable(
                        tf.truncated_normal(shape=[self._vocabulary_size, self._embedding_dimension],
                                            mean=-1.0,
                                            stddev=1.0,
                                            dtype=tf.float64)
                    )
                    embed = tf.map_fn(
                        lambda train_input: tf.nn.embedding_lookup(embeddings, train_input),
                        self._model_input['train'],
                        dtype=tf.float64
                    )
                    embed = tf.transpose(embed, [1, 0, 2, 3])

                with tf.variable_scope('component'):
                    rnn = tf.map_fn(
                        lambda embedding: dynamic_rnn(embedding, cell, state),
                        embed
                    )
                    rnn = tf.reduce_sum(rnn, axis=0)
                    self._model_output['dense'] = dense_encoding(rnn)

                with tf.variable_scope('target'):
                    target_embed = tf.nn.embedding_lookup(embeddings, self._model_input['target'])
                    target_rnn = dynamic_rnn(target_embed, cell, state)
                    matmul = tf.matmul(self._model_output['dense'], tf.transpose(target_rnn))
                    output = tf.boolean_mask(matmul, masking)
                    output = tf.reshape(output, shape=[self._batch_size, 1])
                    self._model_output['sigmoid'] = tf.sigmoid(output)

                # fixme predict 그래프를 따로 그릴 필요성이 없음.
                with tf.variable_scope('predict_embedding_saver'):
                    predict_embeddings = tf.nn.embedding_lookup(embeddings, self._model_input['predict'])
                    predict_rnns = tf.map_fn(
                        lambda predict_embedding: dynamic_rnn(
                            tf.reshape(predict_embedding, shape=[self._batch_size, 30, 300]),
                            cell,
                            state
                        ),
                        predict_embeddings
                    )
                    self._model_output['predict_rnn'] = tf.reshape(predict_rnns, shape=[self._batch_size, self._predict_input_size, 300])

                with tf.variable_scope('predict'):
                    predict_matmul = tf.map_fn(
                        lambda predict_rnn: tf.matmul(self._model_input['dense_e'], tf.transpose(predict_rnn)),
                        self._model_input['saved_title_embedding']
                    )
                    self._model_output['output'] = tf.sigmoid(predict_matmul)

                with tf.variable_scope('optimizer'):
                    positive_reshape = tf.reshape(self._model_input['positive'], [self._batch_size, 1])
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=positive_reshape, logits=output)
                    self._model_output['train'] = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
                    self._model_output['loss'] = tf.reduce_mean(loss)

                init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.graph)
        self.sess.run(init)

    def model_save(self, path):
        path = '/home/wooktae/data/dl_model/kakao/model.ckpt'
        self.saver.save(self.sess, path)

    def model_load(self, path):
        path = '/home/wooktae/data/dl_model/kakao/model.ckpt'
        self.saver.restore(self.sess, path)

    # OOM 발생 ㅠㅠ
    def predict(self, title_name, predict_data):
        predict_result = []
        with open(r'../save_embedding', 'rb') as f:
            embedding_data = pickle.load(f)

        dense_result = self.sess.run(
            self._model_output['dense'],
            feed_dict={self._model_input['train']: predict_data}
        )

        for idx, embedding_datum in enumerate(embedding_data):
            tmp_title_name = title_name[3248 * idx: 3248 * (idx + 1)]
            feed_dict = {
                self._model_input['saved_title_embedding']: embedding_datum,
                self._model_input['dense_e']: dense_result
            }

            predict_scores = self.sess.run(
                self._model_output['output'],
                feed_dict=feed_dict
            )
            zip_data = zip(tmp_title_name, predict_scores[0][0])
            tmp_predict_result = [{'name': x, 'score': y} for x, y in zip_data]
            predict_result += tmp_predict_result

        predict_result = sorted(predict_result, key=lambda x: x.get('score'), reverse=True)
        return predict_result

    def save_input_embedding(self, title_embedding):
        save_embedding = []
        for i in range(198):
            tmp_title_embedding = title_embedding[3248 * i: 3248 * (i + 1)]
            print(i, np.shape(tmp_title_embedding))

            feed_dict = {
                self._model_input['predict']: tmp_title_embedding,
            }

            predict_embedding = self.sess.run(
                self._model_output['predict_rnn'],
                feed_dict=feed_dict
            )
            save_embedding.append(predict_embedding)

        with open(r'../save_embedding', 'wb') as f:
            pickle.dump(save_embedding, f)

    def trainer(self, batch_model: TrainBatch):
        for i in range(10000):
            train_batch, target_batch, positive_batch = batch_model.unpack_batch(self._batch_size)

            feed_dict = {
                self._model_input['train']: train_batch,
                self._model_input['target']: target_batch,
                self._model_input['positive']: positive_batch
            }
            _, result = self.sess.run(
                [self._model_output['train'], self._model_output['loss']],
                feed_dict=feed_dict
            )

            if i % 100 == 0:
                print(result)
                value = self.sess.run(self._model_output['sigmoid'],
                                      feed_dict={self._model_input['train']: train_batch,
                                                 self._model_input['target']: target_batch})
                for p, v in zip(positive_batch, value):
                    print(p, v)


if __name__ == "__main__":
    from pprint import pprint

    ev = EmbeddingVocabulary(r'../tmp/embedding_word')
    mb = MetaDataBatch(r'../tmp/title_metadata', ev)
    devb = DevBatch(r'../tmp/dev', mb)
    train_batch = TrainBatch(r'../tmp/new_train.txt', ev)

    BATCH_SIZE = 100
    PREDICT_INPUT_SIZE = 3248  # mb.max_meta_data_length // 198
    PREDICT_MAX_LENGTH = mb.max_title_length
    VOCABULARY_SIZE = ev.embedding_voca_size

    # pmodel = PopularModel(BATCH_SIZE, PREDICT_INPUT_SIZE, VOCABULARY_SIZE, PREDICT_MAX_LENGTH)
    # pmodel.trainer(train_batch)
    # pmodel.model_save('')

    BATCH_SIZE = 1
    pmodel = PopularModel(BATCH_SIZE, PREDICT_INPUT_SIZE, VOCABULARY_SIZE, PREDICT_MAX_LENGTH)
    pmodel.model_load('')
    dev_data = devb.dev_data
    #pmodel.save_input_embedding(mb.title_embedding)

    with open(r'../dev.recommender.txt', 'a') as f:
        cnt = 0
        for user in open('../tmp/dev.users'):
            k = user
            v = dev_data[user]
            try:
                predict_result = pmodel.predict(mb.title_ids, v)
                result = predict_result[:100]
                result = [x.get('name') for x in result]
                result.insert(0, k)
                result_string = ' '.join(result)
                f.write(result_string+'\n')
            except ValueError as e:
                result_string = " @brunch_151 @sweetannie_145 @chofang1_15 @seochogirl_16 @seochogirl_1 @seochogirl_18 @seochogirl_17 @conbus_43 @seochogirl_11 @hjl0520_26 @seochogirl_12 @seochogirl_13 @seochogirl_14 @seochogirl_15 @wootaiyoung_85 @seochogirl_10 @intlovesong_28 @steven_179 @tenbody_1164 @shindong_38 @tenbody_1305 @seochogirl_8 @seochogirl_7 @shanghaiesther_46 @seochogirl_6 @noey_130 @seochogirl_29 @seochogirl_9 @bzup_281 @seochogirl_2 @seochogirl_3 @hongmilmil_33 @roysday_314 @seochogirl_4 @seochogirl_5 @deckey1985_51 @ohmygod_42 @hyehyodam_19 @boot0715_115 @fuggyee_108 @dailylife_207 @mightysense_9 @syshine7_57 @wikitree_54 @sweetannie_146 @seochogirl_20 @hjl0520_28 @roysday_307 @roysday_313 @13july_92 @brunch_149 @dailylife_219 @aemae-human_15 @ladybob_30 @tamarorim_133 @anetmom_52 @keeuyo_57 @sunnysohn_60 @kidjaydiary_6 @moment-yet_155 @yoriyuri_12 @thebluenile86_4 @ladybob_29 @13july_94 @jijuyeo_9 @namgizaa_46 @anti-essay_150 @thinkaboutlove_234 @dailylife_178 @aemae-human_9 @anetmom_47 @honeytip_945 @hjl0520_27 @moment-yet_157 @dreamwork9_25 @keeuyo_56 @dryjshin_255 @mariandbook_413 @syshine7_56 @boot0715_111 @scienceoflove_5 @seochogirl_28 @everysoso_8 @drunk-traveler_49 @taekangk_61 @ggpodori_14 @dancingsnail_64 @dancingsnail_65 @mentorgrace_8 @juck007_32 @kotatsudiary_66 @studiocroissant_43 @moondol_222 @hum0502_9 @tamarorim_131 @honeytip_841 @dailylife_666 @mothertive_72 @winebee_87 @wonderland_131"
                result_string = str(k) + result_string
                f.write(result_string+'\n')
