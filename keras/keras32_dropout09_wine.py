# 30_9 copy
# mcp save 
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)      # (178, 13) (178,)
print(np.unique(y, return_counts=True))    # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

### one hot encoding ###
y = pd.get_dummies(y)
print(y)
print(y.shape)      # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=512,
                                                    stratify=y)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1)) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4)) 
model.add(Dense(3, activation='softmax'))



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

path = './_save/keras32/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k32_09_', date, '_', filename])    
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
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)
y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
# print("걸린 시간 :", round(end-start,2),'초')



"""
loss : 0.007697440683841705
r2 score : 0.9646814509235723
acc_score : 1.0

[drop out]
loss : 0.0003021926968358457
r2 score : 0.9986181057598148
acc_score : 1.0
"""
