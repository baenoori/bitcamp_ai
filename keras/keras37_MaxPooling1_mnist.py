#35_2 copy

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)  


##### 스케일링 1-1
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.
print(np.max(x_train), np.min(x_train))     # 1.0, 0.0

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#2. 모델 구성 
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28,28,1),
                 strides=1,
                 padding='same'
                 ))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1, padding='same'))     # 24,24,64
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))                         # 23,23,32     
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D())       
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D())       
model.add(Flatten())                                 # 23*23*32

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()


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

path = './_save/keras37/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k37_01_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=128,
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
loss : 0.3148075044155121
acc : 0.91
accuracy_score : 0.8893
걸린 시간 : 4.55 초

loss : 0.03503730148077011
acc : 0.99
accuracy_score : 0.9893
걸린 시간 : 200.69 초

[stide, padding]
loss : 0.03583454340696335
acc : 0.99
accuracy_score : 0.9896
걸린 시간 : 87.41 초

[max pooling]
loss : 0.02903841622173786
acc : 0.99
accuracy_score : 0.9916
걸린 시간 : 75.36 초
"""


