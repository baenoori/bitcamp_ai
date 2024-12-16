#18_1_boston copy

import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston   

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터 
dataset = load_boston()
print(dataset)
print(dataset.DESCR)    # 07.17 추가. describe 확인 
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법

# print(x)
# print(x.shape)  # (506, 13)
# print(y)
# print(y.shape)  # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

#2. 모델 구성
model = Sequential()
# model.add(Dense(32, input_dim=13))
model.add(Dense(32, input_shape=(13,)))      # 이미지 : input_shape=(8,8,1)
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=3, 
          validation_split=0.1
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')

print("=================== hist ==================")
print(hist)

print("================ hist.history =============")
print(hist.history)

print("================ loss =============")
print(hist.history['loss'])
print("================ val_loss =============")
print(hist.history['val_loss'])
print("==================================================")


# import matplotlib.pyplot as plt

# plt.rcParams['font.family'] ='Malgun Gothic'     # 한글 깨짐 해결, 폰트 적용

# plt.figure(figsize=(9,6))   # 9 x 6 사이즈 
# plt.plot(hist.history['loss'],c='red', label='loss',)  # y값 만 넣으면 시간 순으로 그려줌 
# plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')
# plt.legend(loc='upper right')   # 우측 상단 label 표시
# plt.title('보스턴 Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()  # 격자 표시
# plt.show()

""" 
epochs=1000
batch_size=16
loss : 21.718276977539062
r2 score : 0.7094395307207769
걸린 시간 : 17.65 초
"""

