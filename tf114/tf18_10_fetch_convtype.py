import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = pd.get_dummies(y).values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1004,
                                                    stratify=y
                                                    )
print(x_train.shape, y_train.shape) #((435759, 54) (435759, 7)
print(x_test.shape, y_test.shape)   # (145253, 54) (145253, 7)
# exit()

x = tf.placeholder(tf.float32, shape=[None, 54])
y = tf.placeholder(tf.float32, shape=[None, 7])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([54,7], name='weight', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1,7], name='bias', dtype=tf.float32))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

#3-1. 컴파일 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
# print(w_val, b_val)

#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
y_predict = np.argmax(y_predict, 1)
y_data = np.argmax(y_test, 1)

# 정확도 계산
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)  
print('acc :', acc)     # acc : 0.36460520608868663
    
sess.close()