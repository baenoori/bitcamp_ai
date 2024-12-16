# 기존 kaggle 데이터에서 
# 1. train_cav의 y를 casual과 registered로 잡는다.
#    그래서 훈련을 해서 test_cav의 casual과 registered를 predict 한다. 

# 2. test_csv에 casual과 registered 컬럼을 합친다 (파일을 만듦)

# 3. train_csv에 y를 count로 잡는다. 

# 4. 전체 훈련

# 5. test_csv 예측해서 submission에 붙인다. 

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  

#1. 데이터 
path = 'C:/ai5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

#casual과 registered 데이터 예측 후 나누기 
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv[['casual','registered']]

print(x.info())
print(y.columns)   # Index(['casual', 'registered']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2314)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

### csv 파일 ###
y_submit = model.predict(test_csv)
casual_predict = y_submit[:,0]
registered_predict = y_submit[:,1]

print(casual_predict)
test_csv = test_csv.assign(casual=casual_predict, registered = registered_predict)
test_csv.to_csv(path + "test_columnplus.csv")




