import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

######### 실습 #########


########### me ###############

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], name='weight', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일 
loss = -tf.reduce_mean(y * tf.math.log(hypothesis) + (1 - y) * tf.math.log(1 - hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

#3-3. 훈련
epochs = 601
for step in range(epochs):
    _, loss_val, acc_val = sess.run([train, loss, accuracy], feed_dict={x: x_data, y: y_data})
    print(f"Step: {step}  \t  Loss: {loss_val}   \t   Accuracy: {acc_val}")
        
sess.close()



