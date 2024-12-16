# 33_3 copy
# CPU, GPU 시간 체크

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import time

#1. 데이터 
datesets = load_diabetes()
x = datesets.data
y = datesets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=555)

### scaling ###
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
input1 = Input(shape=(10,))
dense1 = Dense(16, activation='relu')(input1)
drop1 = Dropout(0.1)(dense1)
dense2 = Dense(16, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense2)
dense3 = Dense(16, activation='relu')(drop2)
dense4 = Dense(16, activation='relu')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

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
filepath = "".join([path, 'k32_03_', date, '_', filename])  
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
loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
    print('GPU 돈다!~!')
else:
    print('GPU 없다!~!')
    
"""
loss : 3089.77392578125
r2 score : 0.510417644688258

[drop out]
loss : 3138.000732421875
r2 score : 0.502775937187783

[함수형 모델]
loss : 3113.980712890625
r2 score : 0.5065820332231512

[CPU]
loss : 3240.6484375
r2 score : 0.486511112726567
걸린 시간 : 2.37 초
GPU 없다!~!

[GPU]
loss : 3132.83837890625
r2 score : 0.5035939768293065
걸린 시간 : 4.65 초
GPU 돈다!~!
"""

