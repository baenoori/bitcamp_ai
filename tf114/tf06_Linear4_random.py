# 가상환경 : tf114cpu

import tensorflow as tf

#1. 데이터
x = [1,2,3]
y = [1,2,3]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

# w = tf.Variable(tf.random_normal(1), dtype=tf.float32)
# b = tf.Variable(tf.random_normal(1), dtype=tf.float32)
# ValueError: Shape must be rank 1 but is rank 0 for 'random_normal/RandomStandardNormal' (op: 'RandomStandardNormal') with input shapes: [].

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

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
0 0.43976387 [0.53434145] [1.4746451]
20 0.10732773 [0.61950237] [0.8649606]
40 0.040551323 [0.7661171] [0.5316708]
60 0.015321397 [0.85623777] [0.3268054]
80 0.0057888418 [0.9116327] [0.20087957]
100 0.0021871782 [0.9456827] [0.12347595]
'''
