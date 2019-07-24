import tensorflow as tf
from tensorflow import keras
import os


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

