# 배치를 160으로 잡고 x, y 를 추출해서 모델 만들기
# acc 0.99 이상
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical


#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    # horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    # width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    # rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    # fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

test_datagen = ImageDataGenerator(
    rescale=1./255,              # test 데이터는 수치화만!! 
)

path_train = "./_data/image/brain/train/"
path_test = "./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
    path_train,            
    target_size=(200,200),  
    batch_size=160,          
    class_mode='binary',  
    color_mode='grayscale',  
    shuffle=True, 
)

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(200,200),
    batch_size=160,            
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,  
)   

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 200, 200, 1) (120,)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(200,200,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
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

path = './_save/keras41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k41_1_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=160,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test)
r2 = r2_score(y_test,y_pre)
print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)


"""
[IDG]
loss : 0.019881904125213623
acc : 1.0
r2 score : 0.9791341992446339
걸린 시간 : 16.77 초



"""


