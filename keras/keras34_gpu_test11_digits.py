# 33_11 copy
# CPU, GPU 시간 체크 

from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터 
x, y = load_digits(return_X_y=True)     # sklearn에서 데이터를 x,y 로 바로 반환

# print(x)
# print(y)
# print(x.shape, y.shape)     # (1797, 64) (1797,)

print(pd.value_counts(y, sort=False))   # 0~9 순서대로 정렬
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

y_ohe = pd.get_dummies(y)
print(y_ohe.shape)          # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.1, random_state=7777,
                                                    stratify=y)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
input1 = Input(shape=(64,))
dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(64, activation='relu')(drop1)
dense4 = Dense(64, activation='relu')(dense3)
drop2 = Dropout(0.2)(dense4)
dense5 = Dense(64, activation='relu')(drop2)
dense6 = Dense(64, activation='relu')(dense5)
drop3 = Dropout(0.2)(dense6)
dense7 = Dense(64, activation='relu')(drop3)
dense8 = Dense(64, activation='relu')(dense7)
drop4 = Dropout(0.2)(dense8)
dense9 = Dense(64, activation='relu')(drop4)
output1 = Dense(10, activation='softmax')(dense9)
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
print(date)    
print(type(date))  
date = date.strftime("%m%d_%H%M")
print(date)     
print(type(date))  

path = './_save/keras34/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k32_11_', date, '_', filename])     
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
print('loss :',loss[0])
# print('acc :',round(loss[1],2))
print('acc :',loss[1])

y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
# print('걸린 시간 :', round(end-start, 2), '초')
print("걸린 시간 :", round(end-start,2),'초')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
    print('GPU 돈다!~!')
else:
    print('GPU 없다!~!')

"""
loss : 0.005590484477579594
r2 score : 0.9378835045441063
acc_score : 0.9555555555555556

[drop out]
loss : 0.005380750633776188
r2 score : 0.9402138763769592
acc_score : 0.9666666666666667

[함수형 모델]
loss : 0.00898907519876957
r2 score : 0.9001213876469093
acc_score : 0.9444444444444444

[CPU]
loss : 0.004038435406982899
r2 score : 0.9551284962296123
acc_score : 0.9722222222222222
걸린 시간 : 3.38 초
GPU 없다!~!

[GPU]
loss : 0.005989363417029381
r2 score : 0.9334515139632351
acc_score : 0.9666666666666667
걸린 시간 : 8.07 초
GPU 돈다!~!

"""

