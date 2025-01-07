import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

# layer 1 : model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
w1 = tf.compat.v1.Variable(tf.random_normal([2,10], name='weight1'))     # hidden node 10개로
b1 = tf.compat.v1.Variable(tf.zeros([10], name='bias1'))

# hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)
layer1 = tf.matmul(x, w1) + b1

# layer2 : model.add(Dense(5, input_dim=10))
w2 = tf.compat.v1.Variable(tf.random_normal([10,5], name='weight2'))     # hidden node 5개로
b2 = tf.compat.v1.Variable(tf.zeros([5], name='bias2'))
layer2 = tf.matmul(layer1, w2) + b2                                      # layer2의 x는 layer1

# layer3 : model.add(Dense(3))
w3 = tf.compat.v1.Variable(tf.random_normal([5,3], name='weight3'))     
b3 = tf.compat.v1.Variable(tf.zeros([3], name='bias3'))
layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)

# layer4 : model.add(Dense(4))
w4 = tf.compat.v1.Variable(tf.random_normal([3,4], name='weight4'))     
b4 = tf.compat.v1.Variable(tf.zeros([4], name='bias4'))
layer4 = tf.sigmoid(tf.matmul(layer3, w4) + b4)                           

# output_layer : model.add(Dense(1, activation='sigmoid))
w5 = tf.compat.v1.Variable(tf.random_normal([4,1], name='weight5'))     
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bias5'))
hypothesis = tf.sigmoid(tf.matmul(layer4, w5) + b5)

#3-1. 컴파일 
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # binary_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#############
predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))
#############

#3-3. 훈련
epochs = 1001
for step in range(epochs):
    # cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    cost_val, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
        
###################
hypo, pred, acc = sess.run([hypothesis, predict, accuracy], feed_dict={x:x_data, y:y_data})
###################
print("훈련값 :", hypo)
print("예측값 :", pred)
print("acc :", acc)
####################
# print(w_val, b_val)


# # 4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_pre = sess.run(tf.cast(y_predict>0.5, dtype=tf.float32))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pre, y_data)
print('acc :', acc)     # acc : 1.0

