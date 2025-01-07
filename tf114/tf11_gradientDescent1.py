import tensorflow as tf
import matplotlib.pylab as plt

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

#2. 모델
hypothesis = x * w

#3-1. 컴파일        ## model.compile(loss = 'mse', optimizer = 'sgd')
loss = tf.reduce_mean(tf.square(hypothesis-y))

################ 옵티마이저 ################
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
lr = 0.1
# gradient = tf.reduce_mean(2*(x*w - y)*x)  # loss를 w로 미분한 값
gradient = tf.reduce_mean((x*w - y)*x)  # loss를 w로 미분한 값 
descent = w - lr * gradient
train = w.assign(descent)
###########################################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
for step in range(31):
    _, loss_v, w_v = sess.run([train, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()

# print("================================= W history =================================")
# print(w_history)
# print("================================= Loss history =================================")
# print(loss_history)

plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()

