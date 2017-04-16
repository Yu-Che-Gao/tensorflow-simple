import tensorflow as tf
import numpy as np
import lib.nn as nn

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006


def getBatch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS *
                   BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)

    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


if __name__ == '__main__':
    model = nn.LSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE,
                    CELL_SIZE, BATCH_SIZE, LR)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(200):
        seq, res, xs = getBatch()
        if i == 0:
            feedDict = {model.xs: seq, model.ys: res}

        else:
            feedDict = {model.xs: seq, model.ys: res,
                        model.cellInitState: state}

        _, cost, state, pred = sess.run(
            [model.trainOP, model.cost, model.cellFinalState, model.prediction], feed_dict=feedDict)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            returt = sess.run(merged, feedDict)
            writer.add_summary(result, i)
