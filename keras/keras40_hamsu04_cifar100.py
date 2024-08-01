# 함수형 모델

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))   # 0 ~ 99


##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

##### OHE
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

#2. 모델 구성 
input1 = Input(shape=(32,32,3))
dense1 = Conv2D(64, (3,3), padding='same', activation='relu')(input1)
dense2 = Conv2D(64, (3,3), padding='same', activation='relu')(dense1)
dense3 = Conv2D(128, (2,2), padding='same', activation='relu')(dense2)
drop1 = Dropout(0.2)(dense3)
dense5 = Conv2D(256, (2,2), padding='same', activation='relu')(drop1)
maxp1 = MaxPooling2D()(dense5)
dense6 = Conv2D(128, (2,2), padding='same', activation='relu')(maxp1)
drop2 = Dropout(0.2)(dense6)
dense7 = Conv2D(256, (2,2), padding='same', activation='relu')(drop2)


Flat1 = Flatten()(dense7)
drop3 = Dropout(0.7)(Flat1)
dense6 = Dense(256, activation='relu')(drop3)
drop2 = Dropout(0.2)(dense6)
dense7 = Dense(128, activation='relu')(drop2)
output1 = Dense(100, activation='softmax')(dense7)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras40/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k40_04_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=2000, batch_size=64,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측  
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

"""
loss : 2.9738142490386963
acc : 0.27
accuracy_score : 0.273
걸린 시간 : 180.53 초

[BatchNormalization]
loss : 2.424513101577759
acc : 0.38
accuracy_score : 0.3829
걸린 시간 : 173.66 초

[stride, padding]
loss : 2.734889268875122
acc : 0.32
accuracy_score : 0.316
걸린 시간 : 215.21 초

[Max Pooling]
loss : 2.576554775238037
acc : 0.35
accuracy_score : 0.3479
걸린 시간 : 164.28 초

[Max Pooling-함수형]
loss : 2.4728777408599854
acc : 0.37
accuracy_score : 0.373
걸린 시간 : 469.34 초

"""


