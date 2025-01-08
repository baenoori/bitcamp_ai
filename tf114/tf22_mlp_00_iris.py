# 00 ~ 11
# 만들기 
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

y = pd.get_dummies(y).values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1004,
                                                    stratify=y
                                                    )


print(x_train.shape, y_train.shape) # (112, 4) (112, 3)
print(x_test.shape, y_test.shape)   # (38, 4) (38, 3)  

# layer 1 : model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
w1 = tf.compat.v1.Variable(tf.random_normal([4,512], name='weight1'))     # hidden node 10개로
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
w5 = tf.compat.v1.Variable(tf.random_normal([32,3], name='weight5'))     
b5 = tf.compat.v1.Variable(tf.zeros([3], name='bias5'))
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

#3-1. 컴파일 
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 1001
for step in range(epochs):
    # cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    cost_val, _ = sess.run([loss, train], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
        
      
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
y_predict = np.argmax(y_predict, 1)
# y_predict = tf.math.argmax(y_predict, 1)
y_data = np.argmax(y_test, 1)

# 정확도 계산
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)  
print('acc :', acc)     # acc : 0.47368421052631576
      
