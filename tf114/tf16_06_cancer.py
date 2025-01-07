import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1004,
                                                    stratify=y
                                                    )
print(x_train.shape, y_train.shape) # (426, 30) (426,)
print(x_test.shape, y_test.shape)   # (143, 30) (143,)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1], name='weight', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일 
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # binary_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
# print(w_val, b_val)

#4. 평가, 예측
y_predict = tf.sigmoid(tf.matmul(tf.cast(x_test, tf.float32), w_val) + b_val)
y_pre = sess.run(tf.cast(y_predict>0.5, dtype=tf.float32))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pre, y_test)
print('acc :', acc)     # acc : 0.3706293706293706