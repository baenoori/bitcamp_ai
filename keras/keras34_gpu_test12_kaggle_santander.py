# 33_12 copy
# CPU, GPU 시간 체크

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (200000, 200)
print(y.shape)  # (200000,)

print(pd.value_counts(y, sort=True))    # 이진 분류
# 0    179902
# 1     20098

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5233,
                                                    stratify=y)


####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
input1 = Input(shape=(200,))
dense1 = Dense(512, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(512, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(512, activation='relu')(drop2)
dense4 = Dense(512, activation='relu')(dense3)
drop3 = Dropout(0.3)(dense4)
dense5 = Dense(256, activation='relu')(drop3)
dense6 = Dense(128, activation='relu')(dense5)
drop4 = Dropout(0.3)(dense6)
dense7 = Dense(128, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense7)
dense8 = Dense(64, activation='relu')(drop5)
dense9 = Dense(32, activation='relu')(dense8)
dense10 = Dense(16, activation='relu')(dense9)
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
filepath = "".join([path, 'k32_12_', date, '_', filename])  
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1000,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test,y_test,verbose=1)
print('loss :', loss)

y_pre = model.predict(x_test)
r2 = r2_score(y_test,y_pre)
print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
    print('GPU 돈다!~!')
else:
    print('GPU 없다!~!')

# accuracy_score = accuracy_score(y_test,np.round(y_pre))
# print('acc_score :', accuracy_score)
# print('걸린 시간 :', round(end-start, 2), '초')

# ### csv 파일 만들기 ###
# y_submit = model.predict(test_csv)
# print(y_submit)

# y_submit = np.round(y_submit)
# print(y_submit)

# submission_csv['target'] = y_submit
# submission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

# print(submission_csv['target'].value_counts())


"""
loss : 0.10050000250339508
r2 score : -0.1117287381878822

[drop out]
loss : 0.10050000250339508
r2 score : -0.1117287381878822

[함수형 모델]
loss : 0.10050000250339508
r2 score : -0.1117287381878822

[CPU]
loss : 0.10050000250339508
r2 score : -0.1117287381878822
걸린 시간 : 45.95 초
GPU 없다!~!


[GPU]
loss : 0.10050000250339508
r2 score : -0.1117287381878822
걸린 시간 : 8.11 초
GPU 돈다!~!
"""


