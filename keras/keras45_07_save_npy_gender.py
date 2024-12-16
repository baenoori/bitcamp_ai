# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/code

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

#1. 데이터

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

path_train = "C:/ai5/_data/kaggle/biggest gender/faces"

xy_train = train_datagen.flow_from_directory(
    path_train,            
    target_size=(100,100),  
    batch_size=30000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)

# xy_train2 = test_datagen.flow_from_directory(
#     path_train,            
#     target_size=(100,100),  
#     batch_size=30000,          
#     class_mode='binary',  
#     color_mode='rgb',  
#     shuffle=True, 
# )

print(xy_train.class_indices)  # {'man': 0, 'woman': 1}

np_path = "C:/ai5/_data/_save_npy/gender/"
np.save(np_path + 'keras45_07_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_07_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras45_07_x_train2.npy', arr=xy_train2[0][0])
# np.save(np_path + 'keras45_07_y_train2.npy', arr=xy_train2[0][1])

end1 = time.time()
# x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], test_size=0.1, random_state=921)

print('데이터 걸린시간 :',round(end1-start1,2),'초')

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# 데이터 걸린시간 : 48.61 초


