# 가상환경 : tf114cpu

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# tf.set_random_seed(777)
# WARNING:tensorflow:From c:\ai5\study\tf114\tf08_Variable1.py:5: The name tf.set_random_seed is deprecated. Please use tf.compat.v1.set_random_seed instead.
# 잔소리 듣기 싫으면 tf.compat.v1.set_random_seed(777)로 써라

변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weights')
print(변수)     # <tf.Variable 'weights:0' shape=(2,) dtype=float32_ref>

# 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa :', aaa)     # [ 2.2086694  -0.73225045]
sess.close()

# 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)   # 텐서플로 데이터형인 '변수'를 파이썬에서 볼 수 있게 바꿔준다. 
print('bbb :', bbb)     # [ 2.2086694  -0.73225045]
sess.close()

# 초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc :', ccc)     # [ 2.2086694  -0.73225045]
sess.close()
