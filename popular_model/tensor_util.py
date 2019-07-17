import tensorflow as tf


def placeholder(shape, name):
    return tf.placeholder(dtype=tf.float64, shape=shape, name=name)


def lstmcell(output_size, batch_size):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=output_size)
    lstm_state = lstm_cell.zero_state(batch_size=[batch_size], dtype=tf.float64)
    return lstm_cell, lstm_state


def dynamic_rnn(inputs, lstm_cell, lstm_state):
    dy_rnn = tf.nn.dynamic_rnn(
        lstm_cell,
        inputs,
        sequence_length=None,
        initial_state=lstm_state,
        dtype=tf.float64,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False
    )
    return tf.reduce_mean(dy_rnn[0], axis=1)


def dense_encoding(inputs):
    dense_v1 = tf.layers.dense(inputs, 150)
    dense_v2 = tf.layers.dense(dense_v1, 300)
    return dense_v2


def get_gpu_tf_session(graph=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, graph=graph)
    session.run(tf.global_variables_initializer())
    return session
