# DNN -> CNN

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/dacon/diabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]
print(x)    # [652 rows x 8 columns]
print(y.shape)    # (652, )

x = x.to_numpy()
x = x.reshape(652, 4, 2)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=512)


#2. 모델 구성
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(4,2))))
# model.add(Conv2D(128, (3,3), input_shape=(2,2,2), strides=1, activation='relu',padding='same')) 
# model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (3,3), activation='relu', strides=1, padding='same'))        
# model.add(Flatten())                            

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
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

path = './_save/keras59/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k59_07_', date, '_', filename]) 
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
print('loss :', loss[0])
print('acc :', round(loss[1],2))

# print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

# r2 = r2_score(y_test, y_pred)
# print('r2 score :', r2)

y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
# print("걸린 시간 :", round(end-start,2),'초')

print("걸린 시간 :", round(end-start,2),'초')


### csv 파일 ###
# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# # print(y_submit)
# sampleSubmission_csv['Outcome'] = y_submit
# # print(sampleSubmission_csv)
# sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")


"""
loss : 0.17662686109542847
r2 score : 0.23671962825848636
acc_score : 0.7727272727272727

[drop out]
loss : 0.1760784089565277
r2 score : 0.23908975869665205
acc_score : 0.7424242424242424

[함수형 모델]
loss : 0.1697331964969635
r2 score : 0.2665102161316174
acc_score : 0.7575757575757576

[CPU]
loss : 0.18158185482025146
r2 score : 0.21530700200079433
acc_score : 0.7424242424242424
걸린 시간 : 1.43 초
GPU 없다!~!

[GPU]
loss : 0.17883329093456268
r2 score : 0.2271846824795315
acc_score : 0.7424242424242424
걸린 시간 : 4.23 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.5467386245727539
acc : 0.8
acc_score : 0.803030303030303
걸린 시간 : 6.57 초

[lSTM]
loss : 0.5381918549537659
acc : 0.74
acc_score : 0.7424242424242424
걸린 시간 : 8.43 초
"""

