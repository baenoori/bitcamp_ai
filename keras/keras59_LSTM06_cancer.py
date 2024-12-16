# DNN -> CNN

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기 

#1 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)           # 행과 열 개수 확인 
print(datasets.feature_names)   # 열 이름 

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>

# 0과 1의 개수가 몇개인지 찾아보기 
print(np.unique(y, return_counts=True))     # (array([0, 1]), array([212, 357], dtype=int64))

# print(y.value_count)                      # error
print(pd.DataFrame(y).value_counts())       # numpy 인 데이터를 pandas 의 dataframe 으로 바꿔줌
# 1    357
# 0    212
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x = x.reshape(569, 6, 5)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=231)

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)


#2. 모델 구성
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(6,5))))
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
filepath = "".join([path, 'k59_06_', date, '_', filename])  
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



y_pred = model.predict(x_test)
# print(y_pred[:20])

# y_pred = round(y_pred)  # 0 or 1로 acc에 값을 넣기 위해 반올림
# print(y_pred)           # 오류 : y_pred 는 numpy인데 python 함수를 사용하려 해서 오류

y_pred = np.round(y_pred)  # numpy round 함수
# print(y_pred[:20])

from sklearn.metrics import r2_score, accuracy_score    # sklearn 에서 acc 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')



"""
loss : 0.0322132483124733
acc_score : 0.9532163742690059
걸린 시간 : 1.25 초

[drop out]
loss : 0.031158795580267906
acc_score : 0.9590643274853801
걸린 시간 : 1.3 초

[함수형 모델]
loss : 0.024559838697314262
acc_score : 0.9766081871345029
걸린 시간 : 1.87 초

[CPU]
loss : 0.028688447549939156
acc_score : 0.9649122807017544
걸린 시간 : 1.2 초
GPU 없다!~!

[GPU]
loss : 0.03518703952431679
acc_score : 0.9590643274853801
걸린 시간 : 3.94 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.23196353018283844
acc : 0.91
acc_score : 0.9122807017543859
걸린 시간 : 4.3 초

[LSTM]
loss : 0.3226522207260132
acc : 0.89
acc_score : 0.8888888888888888
걸린 시간 : 5.04 초
"""
