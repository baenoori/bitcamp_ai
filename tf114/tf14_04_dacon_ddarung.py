import tensorflow as tf
tf.compat.v1.set_random_seed(777)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

path = "./_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # 점 하나(.) : 루트라는 뜻, index_col=0 : 0번째 열을 index로 취급해달라는 의미
train_csv = train_csv.dropna()
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3333)
print(x_train.shape, y_train.shape) # (1062, 9) (1062,)
print(x_test.shape, y_test.shape)   # (266, 9) (266,)

x = tf.placeholder(tf.float32, shape=[None, 9])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1], name='weight', dtype=tf.float32))
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


# r2_score : -1.8754029227064664e+26
# mae : 1061093858755036.4