import tensorflow as tf


class LSTM(object):
    def __init__(self, nSteps, inputSize, outputSize, cellSize, batchSize, learningRate):
        self.nSteps = nSteps
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.cellSize = cellSize
        self.batchSize = batchSize
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(
                tf.float32, [None, nSteps, inputSize], name='xs')
            self.ys = tf.placeholder(
                tf.float32, [None, nSteps, outputSize], name='ys')

        with tf.name_scope('input-hidden'):
            self.addInputLayer()

        with tf.name_scope('LSTM-cell'):
            self.addCell()

        with tf.name_scope('output-hidden'):
            self.addOutoutLayer()

        with tf.name_scope('cost'):
            self.computeCost()

        with tf.name_scope('train'):
            self.trainOP = tf.train.AdamOptimizer(
                learningRate).minimize(self.cost)

    def addInputLayer(self):
        lInputx = tf.reshape(self.xs, [-1, self.inputSize], name='2_2D')
        WsIn = self._weightVariable(
            [self.inputSize, self.cellSize], name='input-weights')
        bsIn = self._biasVariable([self.cellSize], name='input-biases')

        with tf.name_scope('Ws_plus_b'):
            lInputy = tf.matmul(lInputx, WsIn) + bsIn

        self.lInputy = tf.reshape(
            lInputy, [-1, self.nSteps, self.cellSize], name='2_3D')

    def addCell(self):
        lstmCell = tf.contrib.rnn.BasicLSTMCell(
            self.cellSize, forget_bias=1.0, state_is_tuple=True)

        with tf.name_scope('init-state'):
            self.cellInitState = lstmCell.zero_state(
                self.batchSize, dtype=tf.float32)

        self.cellOutputs, self.cellFinalState = tf.nn.dynamic_rnn(
            lstmCell, self.lInputy, initial_state=self.cellInitState, time_major=False)

    def addOutoutLayer(self):
        lOutputx = tf.reshape(
            self.cellOutputs, [-1, self.cellSize], name='2_2D')
        WsOut = self._weightVariable(
            [self.cellSize, self.outputSize], name='ouput-weights')
        bsOut = self._biasVariable([self.outputSize], name='output-biases')

        with tf.name_scope('Ws_plus_b'):
            self.prediction = tf.matmul(lOutputx, WsOut) + bsOut

    def computeCost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.prediction, [-1], name='reshape-prediction')],
            [tf.reshape(self.ys, [-1], name='reshape-target')],
            [tf.ones([self.batchSize * self.nSteps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.msError,
            name='losses')

        with tf.name_scope('average-cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batchSize,
                name='average-cost')

            tf.summary.scalar('cost', self.cost)

    def msError(self, yPre, yTarget):
        return tf.square(tf.subtract(yPre, yTarget))

    def _weightVariable(self, shape, name='weights'):
        init = tf.random_normal_initializer(mean=0, stddev=1.0)
        return tf.get_variable(shape=shape, name=name, initializer=init)

    def _biasVariable(self, shape, name='biases'):
        init = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=init)
