# DNN -> CNN

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D,MaxPooling2D, Flatten, LSTM, Bidirectional, Conv1D
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

# print(x.shape, y.shape)     # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True))     # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))
# print(pd.value_counts(y, sort=False))
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
# print(y.shape)  # (581012, 7)
# print(y)


x = x.reshape(581012, 9, 6)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5353, stratify=y)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(128, 3, input_shape=(9,6)))
model.add(Flatten())                            

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=7, activation='softmax'))

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

path = './_save/keras60/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k59_10_', date, '_', filename])     
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1000,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))


y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end-start, 2), '초')



"""
loss : 0.013322586193680763
r2 score : 0.8210787449521321
acc_score : 0.9393652542081168
걸린 시간 : 120.93 초

[drop out]
loss : 0.015547509305179119
r2 score : 0.7777253594179163
acc_score : 0.9300024095556091
걸린 시간 : 114.98 초

[함수형 모델]
loss : 0.01559660118073225
r2 score : 0.7729791300948395
acc_score : 0.9278165983959245
걸린 시간 : 111.52 초

[CPU]
loss : 0.01610621064901352
r2 score : 0.7713090457549114
acc_score : 0.9257340539052012
걸린 시간 : 80.07 초
GPU 없다!~!

[GPU]
loss : 0.015298943035304546
r2 score : 0.7836144538524359
acc_score : 0.9308973873532753
걸린 시간 : 165.32 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.3068818747997284
acc : 0.88
r2 score : 0.686575737451579
acc_score : 0.8685759526350212
걸린 시간 : 278.56 초

[LSTM]
loss : 0.12793846428394318
acc : 0.95
r2 score : 0.8663072977526965
acc_score : 0.948211765515817
걸린 시간 : 357.44 초

[Conv1D]
loss : 0.5680022239685059
acc : 0.75
r2 score : 0.41896029362408604
acc_score : 0.7366699941482221
걸린 시간 : 55.76 초


"""



