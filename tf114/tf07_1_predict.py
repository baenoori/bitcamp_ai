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

    # [실습] 예측값 뽑기.
    #4. 예측
    print('===== predict =====')
    x_test_data = [6,7,8]

    # 1. placeholder
    x_test = tf.placeholder(tf.float32, shape=[None])
    # x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test * w_val + b_val
    print(sess.run(y_predict, feed_dict={x_test:x_test_data}))

    # 2. 파이썬 방식
    y_predict2 = x_test_data * w_val + b_val
    print(y_predict2)

'''
[13.000002 15.000002 17.000004]
[13.00000191 15.00000262 17.00000334]

'''
