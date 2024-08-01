# DNN -> CNN
# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/kaggle/playground-series-s4e1/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())

# 문자열 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']


print(x.shape)  # (165034, 10)
print(y.shape)  # (165034,)

x = x.to_numpy()
x = x.reshape(165034, 5, 2, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5324)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(5,2,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu', strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation='relu', strides=1, padding='same'))    
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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

path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k39_08_', date, '_', filename])     
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

# print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

# r2 = r2_score(y_test, y_pred)
# print('r2 score :', r2)
print(y_pred)
y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
# print("걸린 시간 :", round(end-start,2),'초')
print("걸린 시간 :", round(end-start,2),'초')

# ### csv 파일 ###
# y_submit = model.predict(test_csv)

# # print(y_submit)
# y_submit = np.round(y_submit)
# # print(y_submit)
# sampleSubmission_csv['Exited'] = y_submit
# # print(sampleSubmission_csv)
# sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

# print(sampleSubmission_csv['Exited'].value_counts())

"""
loss : 0.10003902018070221
r2 score : 0.3991722537593737
acc_score : 0.8615487154629181

[drop out]
loss : 0.09984055161476135
r2 score : 0.40036401637592434
acc_score : 0.8619728550654386

[함수형 모델]
loss : 0.09969830513000488
r2 score : 0.4012183233479154
acc_score : 0.8611245758603975

[CPU]
loss : 0.09977076947689056
r2 score : 0.4007835079476919
acc_score : 0.8607610276296656
걸린 시간 : 49.8 초
GPU 없다!~!

[GPU]
loss : 0.10007652640342712
r2 score : 0.39894686488398223
acc_score : 0.8614881240911294
걸린 시간 : 267.69 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.36444875597953796
acc : 0.84
acc_score : 0.8416141541444498
걸린 시간 : 191.57 초
"""
