# 가상환경 : tf114cpu

import tensorflow as tf
tf.set_random_seed(777)

#1. 데이터
x = [1,2,3,4,5]
y = [3,5,7,9,11]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# [실습] 만들기

#2. 모델 구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()

'''
0 4501.88 -8.599998 -32.600002
20 74.642654 7.60908 -19.250557
40 37.74071 5.9884405 -13.399535
60 19.082392 4.836054 -9.239054
80 9.64841 4.0166287 -6.2806697
...
920 3.4219739e-12 2.0000012 0.9999957
940 1.6939338e-12 2.0000007 0.99999714
960 1.512035e-12 2.0000007 0.9999976
980 1.512035e-12 2.0000007 0.9999976
1000 1.512035e-12 2.0000007 0.9999976
'''
