# 22_3 copy 
# scaling 추가

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))     # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))
print(pd.value_counts(y, sort=False))
# 5      9493
# 2    283301
# 1    211840
# 7     20510
# 3     35754
# 6     17367
# 4      2747

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2321,
#                                                     stratify=y
#                                                     )

# print(x_train.shape , y_train.shape)    # (522910, 54) (522910,)
# print(x_test.shape , y_test.shape)      # (58102, 54) (58102,)


# print(pd.value_counts(y_train))
# 2    255134
# 1    190623
# 3     32172
# 7     18419
# 6     15542
# 5      8538
# 4      2482


# one hot encoding
y = pd.get_dummies(y)
print(y.shape)  # (581012, 7)
print(y)

# from tensorflow.keras.utils import to_categorical   # keras 이용
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)      # (581012, 8)
# y_ohe = pd.DataFrame(y_ohe)
# print(pd.value_counts(y_ohe, sort=False))


# from sklearn.preprocessing import OneHotEncoder   # sklearn 이용
# y = y.reshape(-1,1) 
# ohe = OneHotEncoder()
# y_ohe = ohe.fit_transform(y)
# print(y_ohe)
# print(y_ohe.shape)      # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5353, stratify=y)

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
model = Sequential()
model.add(Dense(64, input_dim=54, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, batch_size=500,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end-start, 2), '초')


"""
random_state=5353
epochs=1000, batch_size=500
patience=60
loss : 0.20093396306037903
acc : 0.9228
r2 score : 0.7992267483422014
acc_score : 0.92077725379505
걸린 시간 : 466.28 초

stratify=y
loss : 0.1910582333803177
acc : 0.9268
r2 score : 0.8134898619033927
acc_score : 0.9253726205638361
걸린 시간 : 748.42 초

[scaling 추가 - minmax]
loss : 0.15112119913101196
acc : 0.9439
r2 score : 0.8459999030047093
acc_score : 0.9430656431792365
걸린 시간 : 502.74 초

[scaling - standardscaling]
loss : 0.13821472227573395
acc : 0.9488
r2 score : 0.8583522074591432
acc_score : 0.9479880210664005
걸린 시간 : 424.69 초

[scaling - MaxAbsScaler]
loss : 0.15545837581157684
acc : 0.9425
r2 score : 0.8356790360323824
acc_score : 0.9417231764827373
걸린 시간 : 523.04 초

[scaling - RobustScaler]
loss : 0.14241278171539307
acc : 0.9511
r2 score : 0.8597135468682559
acc_score : 0.9505524766789439
걸린 시간 : 342.32 초
"""


