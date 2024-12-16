# 33_2 copy
# CPU, GPU 시간 체크

import sklearn as sk
from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5555)

#### scaling (데이터 전처리)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
input1 = Input(shape=(8,))
dense1 = Dense(16, activation='relu')(input1)
drop1 = Dropout(0.1)(dense1)
dense2 = Dense(16, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense2)
output1 = Dense(1)(drop2)
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
print(date)     # 2024-07-26 16:49:48.004109
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)     # 0726_1655
print(type(date))   # <class 'str'>

path = './_save/keras34/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k32_02_', date, '_', filename])    
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
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

print("걸린 시간 :", round(end-start,2),'초')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
    print('GPU 돈다!~!')
else:
    print('GPU 없다!~!')

"""
loss : 0.29207977652549744
r2 score :  0.7753246532346505

[drop out]
loss : 0.28231775760650635
r2 score :  0.7828337787471844

[함수형 모델]
loss : 0.5426760911941528
r2 score :  0.5825593270822416

[CPU]
loss : 0.31958243250846863
r2 score :  0.7541688811039294
걸린 시간 : 11.01 초
GPU 없다!~!

[GPU]
loss : 0.31930163502693176
r2 score :  0.7543848734863335
걸린 시간 : 126.02 초
GPU 돈다!~!

"""

