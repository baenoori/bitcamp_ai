# DNN -> CNN

import sklearn as sk
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터 
dataset = load_boston()

x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법

print(x.shape)      # (506, 13)

###  reshape, scaling
# x = x.reshape(506,13,1,1)
# y = y.reshape(506,1,1,1)

# x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

####### scaling (데이터 전처리) #######
# x_train = x_train/255.
# x_test = x_test/255.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)  (404, 13) (102, 13)
x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(13,1,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k39_01_', date, '_', filename])    
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

# model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')


#4. 평가, 예측      <- dropout 적용 X
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('loss :', loss[0])
print('acc :', round(loss[1],8))

y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)
# print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')

"""
loss : 10.009329795837402
r2 score : 0.8660890417120943
걸린 시간 : 4.4 초

loss : 9.515417098999023
r2 score : 0.8726969059751946
걸린 시간 : 4.15 초

함수형 모델
loss : 18.923830032348633
r2 score : 0.7468253272484302
걸린 시간 : 2.75 초

CPU
loss : 11.902026176452637
r2 score : 0.8407673794592108
걸린 시간 : 2.67 초
GPU 없다!~!

GPU 
loss : 11.263513565063477
r2 score : 0.849309786922607
걸린 시간 : 5.41 초
GPU 돈다!~!

[DNN -> CNN]
loss : 15.91989517211914
acc : 0.0
걸린 시간 : 7.02 초

"""
