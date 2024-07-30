# 33_7 copy
# CPU, GPU 시간 체크

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
input1 = Input(shape=(8,))
dense1 = Dense(30, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(40, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(50, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(60, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(50, activation='relu')(drop4)
dense6 = Dense(30, activation='relu')(dense5)
drop5 = Dropout(0.1)(dense6)
dense7 = Dense(20, activation='relu')(drop5)
dense8 = Dense(10, activation='relu')(dense7)
dense9 = Dense(7, activation='relu')(dense8)
dense10 = Dense(5, activation='relu')(dense9)
output1 = Dense(1, activation='sigmoid')(dense10)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')

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

path = './_save/keras34/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k32_07_', date, '_', filename]) 
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
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss)
# print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
# print("걸린 시간 :", round(end-start,2),'초')

print("걸린 시간 :", round(end-start,2),'초')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
    print('GPU 돈다!~!')
else:
    print('GPU 없다!~!')

### csv 파일 ###
# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# # print(y_submit)
# sampleSubmission_csv['Outcome'] = y_submit
# # print(sampleSubmission_csv)
# sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")


"""
loss : 0.17662686109542847
r2 score : 0.23671962825848636
acc_score : 0.7727272727272727

[drop out]
loss : 0.1760784089565277
r2 score : 0.23908975869665205
acc_score : 0.7424242424242424

[함수형 모델]
loss : 0.1697331964969635
r2 score : 0.2665102161316174
acc_score : 0.7575757575757576

[CPU]
loss : 0.18158185482025146
r2 score : 0.21530700200079433
acc_score : 0.7424242424242424
걸린 시간 : 1.43 초
GPU 없다!~!

[GPU]
loss : 0.17883329093456268
r2 score : 0.2271846824795315
acc_score : 0.7424242424242424
걸린 시간 : 4.23 초
GPU 돈다!~!
"""

