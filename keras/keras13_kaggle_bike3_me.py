# casual과 registered 을 포함하여 예측

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  


#1. 데이터 
path = 'C:/ai5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test_columnplus.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

print(x.shape)  # (10886, 10)
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5882)

# print(x.info())
# print(y)

#2. 모델 구성
model = Sequential()
model.add(Dense(20, activation='relu', input_dim=10))
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(220, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))


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
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_col_plus_0718_1430.csv")




