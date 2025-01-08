import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
tf.set_random_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)

# layer 1 : model.add(Conv2D(64, (2,2), stride = 1, input_shape=(28, 28, 1)))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 1],
                               initializer=tf.contrib.layers.xavier_initializer()
                               )      # shape=[kernel_size, kernel_size, channel, filter(output)]
b1 = tf.compat.v1.Variable(tf.zeros([1]), name='b1')      # shape=[kernel_size, kernel_size, channel, filter(output)]

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    w1_val = sess.run(w1)
    print(w1_val)
    print(w1_val.shape) # (2, 2, 1, 1)
# [[[[ 0.13086772]]
#   [[-0.62648165]]]
#  [[[-0.0779385 ]]
#   [[ 0.36530012]]]]
