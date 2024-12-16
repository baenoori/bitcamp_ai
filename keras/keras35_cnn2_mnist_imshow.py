import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)    # 특성 부분이 중심쪽에서만 있어서 요약되어 보여져 가장자리 부분의 수치화된 데이터 0만 보임
# print(x_train[0])   # y_train[0]의 데이터와 매칭됨 
# print('y_train[0] :',y_train[0])

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)  28x28 이미지가 60000장 있음, 흑백데이터, 1이 생략되어있음
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)  

print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
print(pd.value_counts(y_test))

import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')  # 데이터 그대로 가져옴
plt.show()



