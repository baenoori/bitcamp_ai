# 32_8 copy
# 함수형 모델
# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/kaggle/playground-series-s4e1/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())

# 문자열 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']


print(x.shape)  # (165034, 10)
print(y.shape)  # (165034,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5324)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
input1 = Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
drop1 = Dropout(0.1)(dense3)
dense4 = Dense(64, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense4)
dense5 = Dense(64, activation='relu')(drop2)
drop3 = Dropout(0.1)(dense5)
dense6 = Dense(64, activation='relu')(drop3)
dense7 = Dense(64, activation='relu')(dense6)
dense8 = Dense(64, activation='relu')(dense7)
drop4 = Dropout(0.1)(dense8)
dense9 = Dense(64, activation='relu')(drop4)
drop5 = Dropout(0.1)(dense9)
dense10 = Dense(64, activation='relu')(drop5)
drop6 = Dropout(0.1)(dense10)
output1 = Dense(1, activation='sigmoid')(drop6)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
print(date)    
print(type(date))  
date = date.strftime("%m%d_%H%M")
print(date)     
print(type(date))  

path = './_save/keras33/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k32_08_', date, '_', filename])     
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss)
# print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)
print(y_pred)
y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
# print("걸린 시간 :", round(end-start,2),'초')


# ### csv 파일 ###
# y_submit = model.predict(test_csv)

# # print(y_submit)
# y_submit = np.round(y_submit)
# # print(y_submit)
# sampleSubmission_csv['Exited'] = y_submit
# # print(sampleSubmission_csv)
# sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

# print(sampleSubmission_csv['Exited'].value_counts())

"""
loss : 0.10003902018070221
r2 score : 0.3991722537593737
acc_score : 0.8615487154629181

[drop out]
loss : 0.09984055161476135
r2 score : 0.40036401637592434
acc_score : 0.8619728550654386

[함수형 모델]
loss : 0.09969830513000488
r2 score : 0.4012183233479154
acc_score : 0.8611245758603975
"""
