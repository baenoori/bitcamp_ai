import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_train = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]
                     ]])
print(x_train.shape)    # (1, 3, 3, 1)

x = tf.placeholder(tf.float32, [None, 3, 3, 1])
w = tf.constant([[[[1.]], [[0.]]],
                  [[[1.]], [[0.]]]])

print(w)        # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)

L1 = tf.nn.conv2d(x, w, strides=(1,1,1,1), padding='VALID')
print(L1)       # Tensor("Conv2D:0", shape=(?, 2, 2, 1), dtype=float32)

sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={x:x_train})
print("===============")
print(output)
# [[[[ 5.]
#    [ 7.]]

#   [[11.]
#    [13.]]]]
print(output.shape)     # (1, 2, 2, 1)
