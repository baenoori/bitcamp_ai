import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  

#1. 데이터
path = 'C:\\ai5\\_data\\bike-sharing-demand\\'   # 절대경로 , 파이썬에서 \\a는 '\a'로 취급 특수문자 쓸 때 주의

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 'datatime'열은 인덱스 취급, 데이터로 X
test2_csv = pd.read_csv(path + "test2.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  # (10886, 11)
print(test2_csv.shape)   # (6493, 10)
print(submission_csv.shape) # (6493, 1)
 
print(train_csv.columns)  

print(train_csv.info()) # 결측치 없음 
print(test2_csv.info())  # 결측치 없음

print(train_csv.describe()) # 합계, 평균, 표준편차, 최소, 최대, 중위값, 사분위값 등을 보여줌 [8 rows x 11 columns]

##### 결측치 확인 #####
print(train_csv.isna().sum())   # 0
print(train_csv.isnull().sum()) # 0

print(test2_csv.isna().sum())    # 0
print(test2_csv.isnull().sum())  # 0

#### x와 y 분리 #####
x = train_csv.drop(['count'], axis = 1)
print(x)    # [10886 rows x 10 columns]
y = train_csv['count']
print(y.shape)    # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=439)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=10))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

### csv 파일 ###
y_submit = model.predict(test2_csv)
print(test2_csv.shape)  # (6493, 1)

submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0718_1320_te.csv")





