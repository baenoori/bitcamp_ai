# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# # ## test 이미지 파일명 변경 ##
# import os
# import natsort

# file_path = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/test"
# file_names = natsort.natsorted(os.listdir(file_path))

# print(np.unique(file_names))
# i = 1
# for name in file_names:
#     src = os.path.join(file_path,name)
#     dst = str(i).zfill(5)+ '.jpg'
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)
#     i += 1

#1. 데이터
path1 = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)

start1 = time.time()
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    # rotation_range=1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'      
)

test_datagen = ImageDataGenerator(
    rescale=1./255,              # test 데이터는 수치화만!! 
)

path_train = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/"
path_test = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/"


xy_train1 = train_datagen.flow_from_directory(
    path_train,            
    target_size=(100,100),  
    batch_size=30000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)

xy_train2 = test_datagen.flow_from_directory(
    path_train,            
    target_size=(100,100),  
    batch_size=30000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(100,100),
    batch_size=30000,            
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,  
)   

print(xy_train1.class_indices)  # {'cat': 0, 'dog': 1}

x = np.concatenate((xy_train1[0][0][:5000],xy_train2[0][0]))
y = np.concatenate((xy_train1[0][1][:5000],xy_train2[0][1]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=921)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')

print(x_train.shape, y_train.shape) # (20000, 100, 100, 3) (20000,)
print(x_test.shape, y_test.shape)   # (5000, 100, 100, 3) (5000,)
# 데이터 걸린시간 : 48.61 초

xy_test = xy_test[0][0]
# print(xy_test)
# print(xy_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

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

path = './_save/keras42/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k42_', date, '_', filename])   
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
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1,
                      batch_size=32
                      )
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test,batch_size=4)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre1 = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre1)
print('accuracy_score :', r2)


### csv 파일 만들기 ###
y_submit = model.predict(xy_test,batch_size=16)
# print(y_submit)

# y_submit = np.round(y_submit,4)
# print(y_submit)

# print(y_submit)
# sampleSubmission_csv['label'] = y_submit
# sampleSubmission_csv.to_csv(path1 + "sampleSubmission_0804_2140.csv")



"""
loss : 0.5235752463340759
acc : 0.7478
걸린 시간 : 132.7 초
accuracy_score : 0.7478
loss : 0.6372641324996948


acc : 0.6278
걸린 시간 : 272.36 초
accuracy_score : 0.6278

loss : 0.6347635388374329
acc : 0.6496
걸린 시간 : 290.02 초
accuracy_score : 0.6496

[0.28]
loss : 0.2779466509819031
acc : 0.88833
걸린 시간 : 304.15 초
accuracy_score : 0.8883333333333333
"""



