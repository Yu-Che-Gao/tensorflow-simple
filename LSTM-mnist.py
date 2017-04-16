import tensorflow as tf
import numpy as np
import lib.nn as nn
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

BATCH_START = 0
STEPS_SIZE = 28
BATCH_SIZE = 128
INPUT_SIZE = 28
OUTPUT_SIZE = 10
CELL_SIZE = 10
LR = 0.001


def getBatch():
    global BATCH_START, STEPS_SIZE
    xs = np.arange(BATCH_START, BATCH_START + STEPS_SIZE *
                   BATCH_SIZE).reshape((BATCH_SIZE, STEPS_SIZE)) / (10 * np.pi)

    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += STEPS_SIZE
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


if __name__ == '__main__':
    model = nn.LSTM(STEPS_SIZE, INPUT_SIZE, OUTPUT_SIZE,
                    CELL_SIZE, BATCH_SIZE, LR)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    step = 0
    while step * BATCH_SIZE < 100000:
        batchXs, batchYs = mnsit.train.next_batch(BATCH_SIZE)
        batchXs = batchXs.reshape([BATCH_SIZE, STEPS_SIZE, INPUT_SIZE])
        sess.run([model.trainOP], feed_dict={
            x: batchXs,
            y: batchYs
        })

        if step % 20 == 0:
            accuracy = tf.reduce_mean(tf.cast(model.prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={
                x: batchXs,
                y: batchYs
            }))

        step += 1

    # for i in range(200):
    #     seq, res, xs = getBatch()
    #     if i == 0:
    #         feedDict = {model.xs: seq, model.ys: res}
    #     else:
    #         feedDict = {model.xs: seq, model.ys: res,
    #                     model.cellInitState: state}

    #     _, cost, state, pred = sess.run(
    #         [model.trainOP, model.cost, model.cellFinalState, model.prediction], feed_dict=feedDict)

    #     if i % 20 == 0:
    #         print('cost: ', round(cost, 4))
    #         result = sess.run(merged, feedDict)
    #         writer.add_summary(result, i)
