import tensorflow as tf
import matplotlib.pylab as plt
tf.set_random_seed(777)

#1. 데이터
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

### [실습] ###
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32, shape=[None])
x3 = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32)

# w1 = tf.compat.v1.Variable([0.001], dtype=tf.float32, name='weight1')
# w2 = tf.compat.v1.Variable([0.003], dtype=tf.float32, name='weight2')
# w3 = tf.compat.v1.Variable([0.004], dtype=tf.float32, name='weight3')

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable([0], dtype=tf.float32, name='bias')

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

w1_history = []
w2_history = []
w3_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 201
for step in range(epochs):
    # _, loss_v, w_v1, w_v2, w_v3 = sess.run([train, loss, w1, w2, w3], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    # print(step, '\t', loss_v, '\t', w_v1, '\t', w_v2, '\t', w_v3)
    coss_val, _ = sess.run([loss, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    print(step, '\t', coss_val)
    
    # w1_history.append(w_v1)
    # w2_history.append(w_v2)
    # w3_history.append(w_v3)
    # loss_history.append(loss_v)
sess.close()

