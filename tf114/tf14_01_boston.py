import tensorflow as tf
tf.compat.v1.set_random_seed(777)

from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, y_train.shape) # (404, 13) (404,)
print(x_test.shape, y_test.shape)   # (102, 13) (102,)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13,1], name='weight', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(loss)

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 51
for step in range(epochs):
    coss_val, _, w_v = sess.run([loss, train, w], feed_dict={x:x_train,y :y_train})
    print(step, '\t', coss_val)
        
sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
y_predict = tf.compat.v1.matmul(tf.cast(x_test, tf.float32), w_v)

with tf.compat.v1.Session() as sess:
    y_pred_val = sess.run(y_predict)

    r2 = r2_score(y_test, y_pred_val)
    mae = mean_absolute_error(y_test, y_pred_val)

    print('r2_score :', r2)
    print('mae :', mae)


# r2_score : -4.09379757978595
# mae : 17.117960204330142