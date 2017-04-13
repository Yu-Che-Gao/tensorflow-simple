# 解 y = 0.1x + 0.3

import tensorflow as tf
import numpy as np

xData = np.random.rand(100).astype(np.float32)
yData = xData * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# create sturctrue
y = Weights * xData + biases
loss = tf.reduce_mean(tf.square(y - yData))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# finish

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
