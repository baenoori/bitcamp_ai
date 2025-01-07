import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# [실습]
# 07_1을 카피해서 아래를 맹그러봐!!!

###### 1. Session() // sess.run(변수) #######
# 블라 블라 블라.....

###### 2. Session() // sess.eval(session=sess) #######
# 블라 블라 블라.....

###### 3. InteractiveSession() // 변수.eval() #######
# 블라 블라 블라.....

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

#2. 모델 구성
hypothesis = x * w + b

# 3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y))  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   # <- Adam쓰고 싶으면 GradientDescentOptimizer을 Adam으로 바꾸면 됨.
train = optimizer.minimize(loss)

# 3-2. 훈련
###### 1. Session() // sess.run(변수) #######
# with tf.compat.v1.Session() as sess:
 #    sess.run(tf.global_variables_initializer()) # 변수 초기화
# 
#     # model.fit()
#     epochs = 1001
#     for step in range(epochs):
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
#                                feed_dict={x: x_data, y: y_data})
#         if step % 20 == 0:
 #              print(f"step{step}, Loss:{loss_val}, W:{w_val}, B:{b_val}")
# 
#     # 4. 예측
#     print("================= predict ==================")
#     x_test = [6,7,8]
 #    
#     y_predict = sess.run(hypothesis, feed_dict={x: x_test})
#     print('[6,7,8]의 예측:', y_predict)
#     sess.close()

# ###### 2. Session() // sess.eval(session=sess) #######
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer()) # 변수 초기화

#     # model.fit()
#    epochs = 1001
#     for step in range(epochs):
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
#                                feed_dict={x: x_data, y: y_data})
#         if step % 20 == 0:
#               print(f"step{step}, Loss:{loss_val}, W:{w_val}, B:{b_val}")
#     # 4. 예측
#     print("================= predict ==================")
#     x_test = [6,7,8]
# 
#     y_predict = hypothesis.eval(feed_dict={x: x_test}, session=sess)
#     print('[6,7,8]의 예측:', y_predict)
#     sess.close()

# ###### 3. InteractiveSession() // 변수.eval() #######
# with tf.compat.v1.InteractiveSession() as sess:
#     sess.run(tf.global_variables_initializer()) # 변수 초기화
# 
#     # model.fit()
#     epochs = 1001
#     for step in range(epochs):
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
#                                feed_dict={x: x_data, y: y_data})
#         if step % 20 == 0:
#               print(f"step{step}, Loss:{loss_val}, W:{w_val}, B:{b_val}")
 #    # 4. 예측
 #    print("================= predict ==================")
#    x_test = [6,7,8]
#     y_predict = hypothesis.eval(feed_dict={x: x_test})
#     print('[6,7,8]의 예측:', y_predict)
#     sess.close()