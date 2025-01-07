# 가상환경 : tf114cpu

import tensorflow as tf

#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델 구성
# y = wx + b -> y = xw + b
hypothesis = x * w + b

#3-1. 컴파일
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

#3-2. 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit()
    epochs = 1001
    for step in range(epochs):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))
    # sess.close()

'''
0 896.29767 8.333321 -44.0       
20 84.863106 11.699302 -24.322027
40 32.06358 7.5766077 -14.950175 
60 12.114484 5.0424848 -9.189518 
80 4.5771785 3.4848197 -5.648579 
100 1.7293819 2.52736 -3.472048
'''
