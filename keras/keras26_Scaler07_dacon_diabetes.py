# 21_1 copy
# scaling 추가
# dacon , 데이터 파일 별도
# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/dacon/diabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]
print(x)    # [652 rows x 8 columns]
print(y.shape)    # (652, )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=512)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=8, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True   
)


hist = model.fit(x_train, y_train, epochs=10000, batch_size=1,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=[es]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')


### csv 파일 ###
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
# print(y_submit)
sampleSubmission_csv['Outcome'] = y_submit
# print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

"""
random_state=512
epochs=1000, batch_size=32

- 1335
loss : 0.19018961489200592
acc : 0.742
r2 score : 0.17810912657023192
acc_score : 0.7424242424242424
걸린 시간 : 1.39 초


test_size=0.1, random_state=512
epochs=10000, batch_size=1
patience=50

loss : 0.21107545495033264
acc : 0.697
r2 score : 0.08785252475280703
acc_score : 0.696969696969697
걸린 시간 : 50.02 초

binary_crossentropy
loss : 0.5893502831459045
acc : 0.773
r2 score : 0.1748338244034645
acc_score : 0.7727272727272727
걸린 시간 : 19.69 초

[scaling 추가 - minmax]
loss : 0.5465951561927795
acc : 0.788
r2 score : 0.24534955069131015
acc_score : 0.7878787878787878
걸린 시간 : 19.7 초

[scaling - Standardscaling]
loss : 0.49388277530670166
acc : 0.803
r2 score : 0.30962293765304694
acc_score : 0.803030303030303
걸린 시간 : 14.41 초

[scaling - MaxAbsScaler]
loss : 0.49517834186553955
acc : 0.803
r2 score : 0.31098507166694067
acc_score : 0.803030303030303
걸린 시간 : 20.91 초

[scaling - RobustScaler]
loss : 0.5882455110549927
acc : 0.727
r2 score : 0.14346713611325268
acc_score : 0.7272727272727273
걸린 시간 : 13.37 초

"""

