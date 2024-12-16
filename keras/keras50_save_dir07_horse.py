# 데이터 증폭화

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#1. 데이터
train_datagen =  ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=0.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

start1 = time.time()
np_path = "c:/ai5/_data/_save_npy/horse/"

x_train = np.load(np_path + 'keras45_02_x_train.npy')
y_train = np.load(np_path + 'keras45_02_y_train.npy')


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=231)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')

print(x_train.shape, y_train.shape) # (821, 200, 200, 3) (821,)
print(x_test.shape, y_test.shape)   # (206, 200, 200, 3) (206,)
# 데이터 걸린시간 : 71.87 초


augment_size = 10000  

randidx = np.random.randint(x_train.shape[0], size = augment_size) 
print(randidx)              
print(np.min(randidx), np.max(randidx)) 

print(x_train[0].shape) 

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   
print(y_augmented.shape)   

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],      
    x_augmented.shape[1],     
    x_augmented.shape[2], 3)    

print(x_augmented.shape)  

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='C:/ai5/_data/_save_img/07_horse/'
).next()[0]

print(x_augmented.shape)   

x_train = x_train.reshape(821, 200, 200, 3)
x_test = x_test.reshape(206, 200, 200, 3)

print(x_train.shape, x_test.shape) 

## numpy에서 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)   

print(np.unique(y_train, return_counts=True))

"""
#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(200,200,3), strides=1, activation='relu',padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(16, (3,3), activation='relu', strides=1, padding='same'))    
model.add(MaxPooling2D())    
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


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
date = date.strftime("%m%d_%H%M")

path = './_save/keras49/7_horse/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k41_4_2_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
"""


"""
loss : 0.16553156077861786
acc : 0.98544
걸린 시간 : 50.05 초
accuracy_score : 0.9854368932038835

[데이터 증폭]
loss : 0.6921750903129578
acc : 0.52913
걸린 시간 : 65.01 초
accuracy_score : 0.529126213592233

[증폭 데이터 세이브(검수)]
loss : 0.6921457052230835
acc : 0.52913
걸린 시간 : 259.69 초
accuracy_score : 0.529126213592233

"""