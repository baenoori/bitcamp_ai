# cat dog 으로 만들기
# image 폴더꺼 수치화, 캐글 폴더 수치화해서 합치고 증폭 1만개 추가해서 만들어서 kaggle에 제출까지

# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM, Bidirectional
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
path1 = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)

start1 = time.time()
"""
# train_datagen =  ImageDataGenerator(
#     # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
#     horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
#     vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
#     width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
#     # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
#     rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
#     # zoom_range=1.2,              # 축소 또는 확대
#     # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
#     fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255,              # test 데이터는 수치화만!! 
# )

# ### image 폴더 cat dog, kaggle cat dog 수치화해서 붙이기 ###
# # kaggle 
# path_train = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/"
# path_test = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/"

# xy_train1 = test_datagen.flow_from_directory(
#     path_train,            
#     target_size=(100,100),  
#     batch_size=30000,          
#     class_mode='binary',  
#     color_mode='rgb',  
#     shuffle=True, 
# )

# xy_test = test_datagen.flow_from_directory(
#     path_test, 
#     target_size=(100,100),
#     batch_size=30000,            
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=False,  
# )   

# path_train = "./_data/image/cat_and_dog/Train/"

# xy_train2 = test_datagen.flow_from_directory(
#     path_train,            
#     target_size=(100,100),  
#     batch_size=20000,          
#     class_mode='binary',  
#     color_mode='rgb',  
#     shuffle=True, 
# )

# xy_test = xy_test[0][0]

# x = np.concatenate((xy_train1[0][0], xy_train2[0][0]))
# y = np.concatenate((xy_train1[0][1], xy_train2[0][1]))

# print(x.shape, y.shape) # (44997, 100, 100, 3) (44997,)
"""

# 데이터 로드 load
train_datagen =  ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

np_path = "C:/ai5/_data/_save_npy/cat_dog_total/"
x = np.load(np_path + 'keras49_05_x_train.npy')
y = np.load(np_path + 'keras49_05_y_train.npy')
xy_test = np.load(np_path + 'keras49_05_x_test.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=921)
end1 = time.time()

print(x_train.shape, y_train.shape) # (40497, 100, 100, 3) (40497,)
print(x_test.shape, y_test.shape)   # (4500, 100, 100, 3) (4500,)

augment_size = 5000

randidx = np.random.randint(x_train.shape[0], size = augment_size) 
x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],      
    x_augmented.shape[1],     
    x_augmented.shape[2], 3)  

x_train = x_train.reshape(40497,100,100,3)
x_test = x_test.reshape(4500,100,100,3)

print(x_train.shape, x_test.shape)  # (40497, 100, 100, 3) (4500, 100, 100, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)   

x_train = x_train.reshape(40497,100,100,3)
x_test = x_test.reshape(4500,100,100,3)

## numpy에서 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)     # (45497, 100, 100, 3) (45497,))

print(np.unique(y_train, return_counts=True))

x_train = x_train.reshape(45497,100,100*3)
x_test = x_test.reshape(4500,100,100*3)

#2. 모델 구성
model = Sequential()
model.add(Bidirectional(LSTM(32, input_shape=(100,100*3), activation='relu', return_sequences=True)))
# model.add(MaxPooling2D())

model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
# model.add(MaxPooling2D())

model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True)))
# model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
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

path = './_save/keras59/5_cat_dog/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k59_05_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=512,
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1,
                      batch_size=2
                      )
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test,batch_size=2)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre1 = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre1)
print('accuracy_score :', r2)


# ### csv 파일 만들기 ###
# y_submit = model.predict(xy_test,batch_size=2)
# # print(y_submit)
# y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))

# print(y_submit)
# sampleSubmission_csv['label'] = y_submit
# sampleSubmission_csv.to_csv(path1 + "sampleSubmission_0806_1750_데이터증폭.csv")



"""
[0.28]
loss : 0.2779466509819031
acc : 0.88833
걸린 시간 : 304.15 초
accuracy_score : 0.8883333333333333

[데이터 ]
loss : 0.2930927872657776
acc : 0.88556
걸린 시간 : 953.75 초
accuracy_score : 0.8855555555555555

[LSTM]
loss : nan
acc : 0.49733
"""




