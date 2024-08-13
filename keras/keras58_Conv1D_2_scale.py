import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Bidirectional, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]              
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 80 맞추기

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)      # (13, 3, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(3,1)))
model.add(Conv1D(64, 2))
model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras58/'
filename = '{epoch:04d}-{loss:.4f}.hdf5' 
filepath = "".join([path, 'k56_2_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(x, y, epochs=5000, batch_size=4, 
        #   validation_split=0.1,
          callbacks=[es, mcp],
          verbose=3
          )

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)

y_pred = model.predict(x_predict.reshape(1,3,1))
print('[50,60,70]의 결과 :', y_pred)    # 80 나오기


# loss : 13.453545570373535
# [50,60,70]의 결과 : [[79.397804]]

# loss : 5.979489803314209
# [50,60,70]의 결과 : [[79.52559]]


##### Bidirectional 사용 #####
# LSTM
# loss : 3.4464609622955322
# [50,60,70]의 결과 : [[66.169685]]

# GRU
# loss : 1.5746080875396729
# [50,60,70]의 결과 : [[67.36417]]

# loss : 18.950786590576172
# [50,60,70]의 결과 : [[79.46454]]

###### Conv1D ###### 
# loss : 7.455232611808249e-12
# [50,60,70]의 결과 : [[80.00001]]


