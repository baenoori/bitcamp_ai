# 함수형 모델

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

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
input1 = Input(shape=(28,28,1))
dense1 = Conv2D(64, (3,3), padding='same', activation='relu')(input1)
dense2 = Conv2D(64, (3,3), padding='same', activation='relu')(dense1)
dense3 = Conv2D(32, (2,2), padding='same', activation='relu')(dense2)
maxp1 = MaxPooling2D()(dense2)
Flat1 = Flatten()(maxp1)
dense4 = Dense(32, activation='relu')(Flat1)
drop1 = Dropout(0.2)(dense4)
dense5 = Dense(16, activation='relu')(drop1)
dense6 = Dense(16, activation='relu')(dense5)
output1 = Dense(10, activation='softmax')(dense6)
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
filepath = "".join([path, 'k40_02_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1, 
          validation_split=0.2,
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
1epo
loss : 0.5463500618934631
acc : 0.81
accuracy_score : 0.8073
걸린 시간 : 5.73 초

loss : 0.2949977517127991
acc : 0.91
accuracy_score : 0.9076z
걸린 시간 : 49.41 초

[stide, padding]
loss : 0.2757048010826111
acc : 0.91
accuracy_score : 0.9092
걸린 시간 : 56.47 초

[Max Pooling]
loss : 0.25262877345085144
acc : 0.92
accuracy_score : 0.916
걸린 시간 : 59.99 초

[Max Pooling-함수형]
loss : 0.2565469443798065
acc : 0.92
accuracy_score : 0.9154
걸린 시간 : 45.35 초

"""
