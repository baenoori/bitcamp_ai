# 가상환경 : tf114cpu
# 07 copy

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

import matplotlib.pyplot as plt
# pip show matplotlib       # Version: 3.1.1

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
w = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

#2. 모델 구성
# y = wx + b -> y = xw + b
hypothesis = x * w + b

#3-1. 컴파일
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
loss_val_list = []
w_val_list = []

# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    # model.fit()
    epochs = 1001
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
    # sess.close()

    # [실습] 예측값 뽑기.
    #4. 예측
    print('===== predict =====')
    x_test_data = [6,7,8]

    # 1. placeholder
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test * w_val + b_val
    print(sess.run(y_predict, feed_dict={x_test:x_test_data}))

    # 2. 파이썬 방식
    y_predict2 = x_test_data * w_val + b_val
    print(y_predict2)

print('========== 그림 그리기 ==========')
print(loss_val_list)
print(w_val_list)

'''
[19.95235, 11.630083, 6.7802567, ..., 8.020029e-06, 7.967292e-06, 7.912307e-06]
[array([1.0985938], dtype=float32), array([1.3244343], dtype=float32), ..., array([2.023233], dtype=float32), array([2.0231543], dtype=float32)]
'''

# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weights')
# plt.grid()
# plt.show()

# plt.plot(w_val_list, loss_val_list)
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# [실습] subplot으로 위 3개의 그래프를 1개로 그리기

plt.subplot(221)
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.subplot(222)
plt.plot(w_val_list)
plt.xlabel('epochs')
plt.ylabel('weights')
plt.grid()

plt.subplot(223)
plt.plot(w_val_list, loss_val_list)
plt.xlabel('weights')
plt.ylabel('loss')
plt.grid()

plt.show()
