 # import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk
from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) # (2040, 8) (20640, )

#[실습] 만들기
# R2 0.59 이상

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=189)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=40)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

""" 
test_size : 0.2
random_state : 189
epo : 500
batch_size : 40
loss : 0.6045974493026733
r2 score :  0.5415096393393787
"""
