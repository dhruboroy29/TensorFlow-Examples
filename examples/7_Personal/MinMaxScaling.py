import numpy as np
import tensorflow as tf

x = tf.placeholder(dtype=tf.float32)

max_x = tf.reduce_max(x, axis=0)
min_x = -1 * tf.reduce_max(-1 * x, axis=0)
scaled_x = (x-min_x)

with tf.Session() as sess:
    arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print('Max, Min_x, Scaled array: ', sess.run([max_x, min_x, scaled_x], {x:arr}))