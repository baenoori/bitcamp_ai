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

np_path = "c:/ai5/_data/_save_npy/brain/"
np.save(np_path + 'keras45_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_01_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras45_01_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras45_01_y_test.npy', arr=xy_test[0][1])

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
