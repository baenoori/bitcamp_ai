import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[77, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]
          ]
y_data = [[152], [185], [180], [205], [142]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bias'))

##### [실습] #####

#2. 모델
# hypothesis = x*w + b
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 51
for step in range(epochs):
    coss_val, _ = sess.run([loss, train], feed_dict={x:x_data,y :y_data})
    print(step, '\t', coss_val)
        
sess.close()
