#11_1 boston copy
# verbose, validation 추가 

import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston    # 현 버전에서는 boston 데이터셋 사용 가능함 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터 
dataset = load_boston()
print(dataset)
print(dataset.DESCR)    # 07.17 추가. describe 확인 
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법

print(x)
print(x.shape)  # (506, 13)
print(y)
print(y.shape)  # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=0,            # 추가
          validation_split=0.1  # 추가
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

""" 
test_size : 0.2
random_state : 123
epo : 1000
batch_size : 16
loss : 21.67367935180664
r2 score : 0.7510105661569786

<val 추가>
validation_split=0.1
loss : 21.08422088623047
r2 score : 0.7179223398628665
"""
