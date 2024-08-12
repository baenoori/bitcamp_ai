# Conv2D로 시작해서 중간에 LSTM 을 넣어 모델 구성하기

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D, LSTM
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

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 10)

#2. 모델 구성 
model = Sequential()
model.add(Dense(28, input_shape=(28,28)))   # (N, 28, 28)
model.add(Reshape(target_shape=(28,28,1)))  # (N, 28, 28, 1)   # target shape 은 input shape와 동일하게 

model.add(Conv2D(64, (3,3)))  # 26,26,64
model.add(MaxPooling2D())     # 13,13,64
model.add(Conv2D(5, (4,4)))  # 10,10,100

model.add(Reshape(target_shape=(10*10, 5)))    # (100,5) 
model.add(LSTM(10, return_sequences=True))

model.add(Flatten())    # (500,)

model.add(Dense(10, activation='relu'))
model.summary()


# """
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras57/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k35_04_', date, '_', filename])   
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
# """


"""
1epo
loss : 0.3148075044155121
acc : 0.91
accuracy_score : 0.8893
걸린 시간 : 4.55 초

loss : 0.3178982734680176
acc : 0.91
accuracy_score : 0.9121
걸린 시간 : 4.96 초

loss : 0.03503730148077011
acc : 0.99
accuracy_score : 0.9893
걸린 시간 : 200.69 초

reshape
loss : nan
acc : 0.1
accuracy_score : 0.098
걸린 시간 : 320.31 초


"""










