# 30_10 copy
# mcp load

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
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
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(64, input_dim=54, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(7, activation='softmax'))


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10, verbose=1,
#                    restore_best_weights=True,
#                    )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# print(date)    
# print(type(date))  
# date = date.strftime("%m%d_%H%M")
# print(date)     
# print(type(date))  

# path = 'C:/ai5/_save/keras30_mcp/10_fetch_covtype/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    
# filepath = "".join([path, 'k30_', date, '_', filename])    
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=1000,
#           verbose=1, 
#           validation_split=0.1,
#           callbacks=[es, mcp],
#           )
# end = time.time()

#4. 평가, 예측
print("============================ MCP 출력 ==============================")
model2 = load_model('C:/ai5/_save/keras30_mcp/10_fetch_covtype/k30_0726_2022_0117-0.0133.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)


"""
loss : 0.013322586193680763
r2 score : 0.8210787449521321
acc_score : 0.9393652542081168
걸린 시간 : 120.93 초


loss : 0.013322586193680763
r2 score : 0.8210787449521321

"""


