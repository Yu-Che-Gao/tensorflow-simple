import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


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


def computeAccuracy(vXs, vYs):
    global prediction
    yPre = sess.run(prediction, feed_dict={xs: vXs})
    correctPrediction = tf.equal(tf.argmax(yPre, 1), tf.argmax(vYs, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: vXs, ys: vYs})
    return result


# Define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = addLayer(xs, 784, 10, activationFunc=tf.nn.softmax)

# the error between prediction and real data
crossEntropy = tf.reduce_mean(-tf.reduce_sum(ys *
                                             tf.log(prediction), reduction_indices=[1]))

trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batchXs, batchYs = mnist.train.next_batch(100)
    sess.run(trainStep, feed_dict={xs: batchXs, ys: batchYs})
    if i % 50 == 0:
        print(computeAccuracy(mnist.test.images, mnist.test.labels))
