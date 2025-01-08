import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
tf.set_random_seed(777)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64],
                               initializer=tf.contrib.layers.xavier_initializer()
                               )      # shape=[kernel_size, kernel_size, channel, filter(output)]
b1 = tf.compat.v1.Variable(tf.zeros([64]), name='b1')      # shape=[kernel_size, kernel_size, channel, filter(output)]

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 += b1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 64), dtype=float32)


w2 = tf.compat.v1.get_variable('w2', shape=[2, 2, 64, 32],
                               initializer=tf.contrib.layers.xavier_initializer()
                               )      # shape=[kernel_size, kernel_size, channel, filter(output)]
b2 = tf.compat.v1.Variable(tf.zeros([32]), name='b2')      # shape=[kernel_size, kernel_size, channel, filter(output)]

L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 += b2
L2 = tf.nn.relu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(?, 7, 7, 32), dtype=float32)

L_flat = tf.reshape(L2_maxpool, [-1, 7*7*32])
print(L_flat)   # Tensor("Reshape:0", shape=(?, 1568), dtype=float32)

w3 = tf.compat.v1.get_variable(shape=[1568,64], name='weight3', initializer=tf.contrib.layers.xavier_initializer())  
b3 = tf.compat.v1.Variable(tf.zeros([64], name='bias3'))
layer3 = tf.nn.relu(tf.matmul(L_flat, w3) + b3)
print(layer3)

# layer4 : model.add(Dense(4))
w4 = tf.compat.v1.Variable(tf.random_normal([64,32], name='weight4'))     
b4 = tf.compat.v1.Variable(tf.zeros([32], name='bias4'))
layer4 = tf.sigmoid(tf.matmul(layer3, w4) + b4)                           

# output_layer : model.add(Dense(1, activation='sigmoid))
w5 = tf.compat.v1.Variable(tf.random_normal([32,10], name='weight5'))     
b5 = tf.compat.v1.Variable(tf.zeros([10], name='bias5'))
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

#3-1. 컴파일 
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 51
batch_size = 10000
total_batch = int(len(x_train) / batch_size)        # 600

for step in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        
        # cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
        cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4], feed_dict=feed_dict)
        
        avg_cost += cost_val
    avg_cost /= total_batch
        
    if step % 2 == 0:
        print(step, 'loss :', avg_cost)
      
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
y_predict = np.argmax(y_predict, 1)
# y_predict = tf.math.argmax(y_predict, 1)
y_data = np.argmax(y_test, 1)

# 정확도 계산
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)  
print('acc :', acc)     # acc : 0.893


