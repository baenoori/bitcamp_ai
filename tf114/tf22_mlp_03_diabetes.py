import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3333)
print(x_train.shape, y_train.shape) # (353, 10) (353,)
print(x_test.shape, y_test.shape)   # (89, 10) (89,)


# layer 1 : model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,])
w1 = tf.compat.v1.Variable(tf.random_normal([10,512], name='weight1'))     # hidden node 10개로
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
layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)                           

# output_layer : model.add(Dense(1, activation='sigmoid))
w5 = tf.compat.v1.Variable(tf.random_normal([32,1], name='weight5'))     
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bias5'))
hypothesis = tf.matmul(layer4, w5) + b5

#3-1. 컴파일 
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(tf.square(hypothesis - y)) 

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 201
for step in range(epochs):
    # cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    cost_val, _ = sess.run([loss, train], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
        
      
y_predict = sess.run(hypothesis, feed_dict={x:x_test})

from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

print('r2_score :', r2) # r2_score : -0.2112237704348221
print('mae :', mae)     # mae : 65.68856297182234
