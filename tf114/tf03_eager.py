import tensorflow as tf
print('tf version :', tf.__version__)   # tf version : 1.14.0
print('즉시실행모드 :', tf.executing_eagerly()) # 즉시실행모드 : False

# 가상환경 변경 : tf274cpu 
print('tf version :', tf.__version__)   # tf version : 2.7.4 
print('즉시실행모드 :', tf.executing_eagerly()) # 즉시실행모드 : True

tf.compat.v1.disable_eager_execution()   # 즉시 실행 모드 끄기 
print('즉시실행모드 :', tf.executing_eagerly()) # 즉시실행모드 : False

tf.compat.v1.enable_eager_execution()    # 즉시 실행 모드 킴
print('즉시실행모드 :', tf.executing_eagerly()) # 즉시실행모드 : True

# 쯕시 실행모드 -> 텐서1의 그래프 형태의 구성 없이 자연스러운 파이썬 문법으로 실행
# tf.compat.v1.disable_eager_execution()   # 즉시 실행 모드 끄기 // 텐서플로우 1.0 문법 (디폴트
# tf.compat.v1.enable_eager_execution()    # 즉시 실행 모드 킴  // 텐서플로우 2.0 사용 가능

hello = tf.constant('Hello world!')
sess = tf.compat.v1.Session()

print(sess.run(hello))  # b'Hello world!'

#  가상환경         즉시 실행 모드              사용가능
#   1.14.0          disable (디폴트)           b'Hello world!'
#   1.14.0          enable                    error
#   2.7.4           disable (디폴트)           b'Hello world!'
#   2.7.4           enable                    error

"""
Tensor1 은 '그래프 연산' 모드
Tensor2 는 '즉시 실행' 모드

tf.compat.v1.enable_eager_execution()    # 즉시 실행 모드 킴
-> Tensor 2의 디폴트

tf.compat.v1.disable_eager_execution()   # 즉시 실행 모드 끄기 
-> 그래프 연산모드로 돌아감
-> Tensor 1코드를 쓸 수 있음

tf.executing_eagerly()
-> True : 즉시 실행 모드, Tensor 2 코드만 써야함
-> False : 그래프 연산 모드, Tensor 1 코드를 쓸 수 있음
"""


