import tensorflow as tf
tf.compat.v1.set_random_seed(777)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3333)
print(x_train.shape, y_train.shape) # (353, 10) (353,)
print(x_test.shape, y_test.shape)   # (89, 10) (89,)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weight', dtype=tf.float32))
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
epochs = 21
for step in range(epochs):
    coss_val, _, w_v = sess.run([loss, train, w], feed_dict={x:x_train,y :y_train})
    print(step, '\t', coss_val)
        
sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
y_predict = tf.compat.v1.matmul(tf.cast(x_test, tf.float32), w_v)

# 이후 세션을 열고 y_predict를 실행
with tf.compat.v1.Session() as sess:
    y_pred_val = sess.run(y_predict)

    # sklearn 메트릭 계산
    r2 = r2_score(y_test, y_pred_val)
    mae = mean_absolute_error(y_test, y_pred_val)

    print('r2_score :', r2)
    print('mae :', mae)


# r2_score : -5.089171150861544
# mae : 167.7994641893899