# cat dog 으로 만들기

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# #1. 데이터
# start1 = time.time()
# train_datagen = ImageDataGenerator(
#     rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
#     # horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
#     # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
#     # width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
#     # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
#     # rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
#     # zoom_range=1.2,              # 축소 또는 확대
#     # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
#     # fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255,              # test 데이터는 수치화만!! 
# )

# path_train = "./_data/image/horse_human/"
# path_test = "./_data/image/horse_human/"

# xy_train = train_datagen.flow_from_directory(
#     path_train,            
#     target_size=(200,200),  
#     batch_size=20000,          
#     class_mode='binary',  
#     color_mode='rgb',  
#     shuffle=True, 
# )

# xy_test = test_datagen.flow_from_directory(
#     path_test, 
#     target_size=(200,200),
#     batch_size=20000,            
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,  
# )   


# x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], test_size=0.2, random_state=231)
# end1 = time.time()

# print('데이터 걸린시간 :',round(end1-start1,2),'초')

# print(x_train.shape, y_train.shape) # (821, 100, 100, 3) (821,)
# print(x_test.shape, y_test.shape)   # (206, 100, 100, 3) (206,)
# # 데이터 걸린시간 : 71.87 초


# #2. 모델 구성
# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape=(200,200,3), strides=1, activation='relu',padding='same')) 
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (3,3), activation='relu', strides=1, padding='same'))    
# model.add(MaxPooling2D())    
# model.add(Flatten())
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10, verbose=1,
#                    restore_best_weights=True,
#                    )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# path = './_save/keras41/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
# filepath = "".join([path, 'k41_4_', date, '_', filename])   
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
#           validation_split=0.1,
#           callbacks=[es, mcp],
#           )
# end = time.time()

#4. 평가, 예측
np_path = "c:/ai5/_data/_save_npy/horse/"


x_train = np.load(np_path + 'keras45_02_x_train.npy')
y_train = np.load(np_path + 'keras45_02_y_train.npy')
x_test = np.load(np_path + 'keras45_02_x_test.npy')
y_test = np.load(np_path + 'keras45_02_y_test.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=231)

print("============================ MCP 출력 ==============================")
model2 = load_model('C:/ai5/_save/keras41/k41_4_0802_1505_0085-0.1410_horse.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)


"""
loss : 0.16553156077861786
acc : 0.98544
걸린 시간 : 50.05 초
accuracy_score : 0.9854368932038835

[load_data]
loss : [0.15201780200004578, 1.0]
r2 score : 0.8669982070656288

"""