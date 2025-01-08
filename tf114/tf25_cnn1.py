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
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64])      # shape=[kernel_size, kernel_size, channel, filter(output)]

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')
# stride = 2 일 경우에는 [1,2,2,1]

print(w1)       # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)       # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)

# layer 2:
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 64, 32])      # shape=[kernel_size, kernel_size, channel, filter(output)]
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='VALID')

print(w2)       # <tf.Variable 'w2:0' shape=(3, 3, 64, 32) dtype=float32_ref>
print(L2)       # Tensor("Conv2D_1:0", shape=(?, 13, 13, 32), dtype=float32)
