import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
path = "C:/ai5/_data/kaggle/playground-series-s4e1/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)[:20000]
y = train_csv['Exited'][:20000]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1004,
                                                    stratify=y
                                                    )
print(x_train.shape, y_train.shape) # (123775, 10) (123775,)
print(x_test.shape, y_test.shape)   # (41259, 10) (41259,) 

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weight', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일 
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # binary_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-3. 훈련
epochs = 101
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
print('acc :', acc)     # acc : 0.7932
    