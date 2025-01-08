import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
tf.set_random_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

print(x_train.shape, y_train.shape) # (60000, 784) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 784) (10000, 10)

#### 실습 #### 

# layer 1 : model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
w1 = tf.compat.v1.get_variable(shape=[784,512], name='weight1', initializer=tf.contrib.layers.xavier_initializer())     # hidden node 10개로
b1 = tf.compat.v1.Variable(tf.zeros([512], name='bias1'))

# hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)
layer1 = tf.matmul(x, w1) + b1


##################### 드랍아웃 적용 #####################
# layer2 : model.add(Dense(5, input_dim=10))
w2 = tf.compat.v1.get_variable(shape=[512,128], name='weight2', initializer=tf.contrib.layers.xavier_initializer())   # hidden node 5개로
b2 = tf.compat.v1.Variable(tf.zeros([128], name='bias2'))
layer2 = tf.matmul(layer1, w2) + b2                                      # layer2의 x는 layer1

layer2 = tf.nn.dropout(layer2, rate=0.3)
#######################################################

###################### relu ######################
# layer3 : model.add(Dense(3))
w3 = tf.compat.v1.Variable(tf.random_normal([128,64], name='weight3'))     
b3 = tf.compat.v1.Variable(tf.zeros([64], name='bias3'))
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
# layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3)
# layer3 = tf.nn.leaky_relu(tf.matmul(layer2, w3) + b3)
# layer3 = tf.nn.elu(tf.matmul(layer2, w3) + b3)
#################################################

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

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 201
batch_size = 100
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
        
    if step % 20 == 0:
        print(step, 'loss :', avg_cost)
    
      
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
y_predict = np.argmax(y_predict, 1)
# y_predict = tf.math.argmax(y_predict, 1)
y_data = np.argmax(y_test, 1)

# 정확도 계산
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)  
print('acc :', acc)     # acc : 0.1225
      