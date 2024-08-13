# DNN -> CNN

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  
import time

#1. 데이터
# path = "./_data/따릉이/"    # 상대경로 
path = 'C:/ai5/_data/kaggle/bike-sharing-demand/'   # 절대경로 , 파이썬에서 \\a는 '\a'로 취급 특수문자 쓸 때 주의
# path = 'C:/ai5/_data/bike-sharing-demand'       # 위와 동일, //도 가능 

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 'datatime'열은 인덱스 취급, 데이터로 X
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)  # (10886, 11)
# print(test_csv.shape)   # (6493, 8)
# print(submission_csv.shape) # (6493, 1)
 
# print(train_csv.columns)  
# # Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')

# print(train_csv.info()) # 결측치 없음 
# print(test_csv.info())  # 결측치 없음

# print(train_csv.describe()) # 합계, 평균, 표준편차, 최소, 최대, 중위값, 사분위값 등을 보여줌 [8 rows x 11 columns]

##### 결측치 확인 #####
# print(train_csv.isna().sum())   # 0
# print(train_csv.isnull().sum()) # 0

# print(test_csv.isna().sum())    # 0
# print(test_csv.isnull().sum())  # 0

#### x와 y 분리 #####
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [0, 0] < list (2개 이상은 리스트)
print(x)    # [10886 rows x 8 columns]
y = train_csv['count']
print(y)

x = x.to_numpy()

x = x.reshape(10886, 4, 2)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=654)


#2. 모델 구성
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(4,2))))
# model.add(Conv2D(128, (3,3), input_shape=(2,2,2), strides=1, activation='relu',padding='same')) 
# model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
model.add(MaxPooling2D())
# model.add(Conv2D(64, (3,3), activation='relu', strides=1, padding='same'))        
# model.add(Flatten())                            

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

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

path = './_save/keras59/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k59_05_', date, '_', filename])  
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :', round(loss[1],5))


print("걸린 시간 :", round(end-start,2),'초')

# print("걸린 시간 :", round(end-start),'초')
# # ### csv 파일 ###
# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# submission_csv.to_csv(path + "submission_0725_1720_RS.csv")


"""
loss : 21463.5078125
r2 : 0.3185081846417004

[drop out]
loss : 21207.0703125
r2 : 0.3266504356964226

[함수형 모델]
loss : 21340.126953125
r2 : 0.3224257254081804

[CPU]
loss : 21711.5234375
r2 : 0.3106333787818707
걸린 시간 : 14.78 초
GPU 없다!~!

[GPU]
loss : 21256.560546875
r2 : 0.3250789782997482
걸린 시간 : 78.66 초
GPU 돈다!~!

[DNN -> CNN]
loss : 21605.263671875
acc : 0.00826
걸린 시간 : 68.66 초

[LSTM]
loss : 3018.698974609375
acc : 0.0
걸린 시간 : 17.43 초
"""

