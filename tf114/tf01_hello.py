import tensorflow as tf
print(tf.__version__)       # 1.14.0

## 텐서플로 설치 오류시
# pip install protobuf==3.20
# pip install numpy==1.16

print('hello world')

hello = tf.constant('hello world')
print(hello)            # Tensor("Const:0", shape=(), dtype=string)
# tensor1 머신에 들어갔을 때 그래프의 상태를 출력, 결과값 x
# tensor machine 의 상태

sess = tf.Session()
print(sess.run(hello))       # 그래프 연산을 실행 시킴
# b'hello world'
