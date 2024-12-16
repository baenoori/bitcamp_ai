# 과적합 방지를 위해 훈련을 할 때마다 임의의 노드를 제거 -> drop out 
# 29_5 copy

import sklearn as sk
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Conv1D, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터 
dataset = load_boston()

x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=231)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(13,1))) # timesteps , features
model.add(Dropout(0.3))     # 64개의 30% 를 제외하고 훈련. 상위 레이어에 종속적 
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
print(date)     # 2024-07-26 16:49:48.004109
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)     # 0726_1655
print(type(date))   # <class 'str'>

path = './_save/keras60/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k60_01_', date, '_', filename])    
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
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')


# loss : 10.009329795837402
# r2 score : 0.8660890417120943
# 걸린 시간 : 4.4 초

# loss : 9.515417098999023
# r2 score : 0.8726969059751946
# 걸린 시간 : 4.15 초

### LSTM ###
# loss : 24.527427673339844
# r2 score : 0.5924775870886514
# 걸린 시간 : 14.94 초

### Conv1D ###
# loss : 14.606295585632324
# r2 score : 0.7573168653214845
# 걸린 시간 : 5.9 초