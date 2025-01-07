# 가상환경 : tf114cpu

import tensorflow as tf

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

#2. 모델 구성
# y = wx + b -> y = xw + b
hypothesis = x * w + b

#3-1. 컴파일
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

#3-2. 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit()
    epochs = 1001
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)
    # sess.close()

'''
0 18.637705 [1.9318104] [2.102371]
20 0.10326425 [1.7948985] [1.7404814]
40 0.05221234 [1.8541589] [1.5265329]
60 0.026399542 [1.896297] [1.3744009]
80 0.013348049 [1.9262601] [1.2662245]
100 0.006749002 [1.9475659] [1.1893038]
'''
