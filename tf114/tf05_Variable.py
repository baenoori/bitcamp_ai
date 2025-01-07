# 가상환경 : tf114cpu

import tensorflow as tf
sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([3], dtype=tf.float32)
# a = 2, b = 3

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(a + b))  # [5.]

# init = tf.compat.v1.global_variables_initializer()
# sess.run(init)

# print(sess.run(a + b))
# tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Variable [[{{node Variable/read}}]]
