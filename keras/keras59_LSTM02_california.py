# 30_2 copy
# drop out

import sklearn as sk
from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM,Flatten
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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(8,1)))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=5, verbose=1,
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

path = './_save/keras59/'
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


"""
loss : 0.29207977652549744
r2 score :  0.7753246532346505

[drop out]
loss : 0.28231775760650635
r2 score :  0.7828337787471844

[LSTM]
loss : 0.3826203942298889
r2 score :  0.7056784020616906

"""

