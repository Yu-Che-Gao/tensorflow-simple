import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def addLayer(inputs, inSize, outSize, activationFunc=None, layerName='defaultLayer'):
    with tf.name_scope(layerName):
        inputs = tf.cast(inputs, dtype=tf.float32)
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal(
                [inSize, outSize]), dtype=tf.float32, name='W')
            tf.summary.histogram('weights', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, outSize]) + 0.1, name='b')
            tf.summary.histogram('biases', biases)

        with tf.name_scope('WxPlus_b'):
            WxPlus_b = tf.matmul(inputs, Weights) + biases

        if activationFunc is None:
            outputs = WxPlus_b  # linear relationship
        else:
            outputs = activationFunc(WxPlus_b)

        tf.summary.histogram('outputs', outputs)

        return outputs


xData = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, xData.shape)
yData = np.square(xData) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='xInput')
    ys = tf.placeholder(tf.float32, [None, 1], name='yInput')

l1 = addLayer(xData, 1, 10, activationFunc=tf.nn.relu, layerName='layer1')
prediction = addLayer(l1, 10, 1, activationFunc=None, layerName='layer2')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(yData - prediction), reduction_indices=[1]))  # 平均誤差
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    trainStep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 減小誤差

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/', sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(trainStep, feed_dict={xs: xData, ys: yData})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: xData, ys: yData})
        writer.add_summary(result, i)