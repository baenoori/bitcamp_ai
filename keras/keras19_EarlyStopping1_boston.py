#18_1_boston copy
# early stopping 

import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston    # 현 버전에서는 boston 데이터셋 사용 가능함 

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
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',       # 기준
    mode = 'min',               # 최소값 찾기, 모르면 auto(loss등는 자동으로 최소값, acc/r2등은 자동으로 최대값으로)
    patience = 10,              # patience 동안 갱신되지 않으면 훈련을 끝냄
    restore_best_weights=True,  # 최소값인 지점의 weight를 최종 가중치로 잡음, 쓰지 않으면 최소값 지점이 아닌 훈련이 끝나는 지점의 weight 값을 최종 가중치로 잡음
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es],       # EarlyStopping 적용
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


import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'     # 한글 깨짐 해결, 폰트 적용

plt.figure(figsize=(9,6))   # 9 x 6 사이즈 
plt.plot(hist.history['loss'],c='red', label='loss',)  # y값 만 넣으면 시간 순으로 그려줌 
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')
plt.legend(loc='upper right')   # 우측 상단 label 표시
plt.title('보스턴 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시
plt.show()

""" 
epochs=1000
batch_size=16
loss : 21.718276977539062
r2 score : 0.7094395307207769
걸린 시간 : 17.65 초
"""
