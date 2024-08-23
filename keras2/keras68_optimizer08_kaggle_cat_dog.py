# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling1D, BatchNormalization, LSTM, Bidirectional, Conv1D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
path1 = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)

start1 = time.time()

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=921)
end1 = time.time()

print(x_train.shape, y_train.shape) # (22498, 100, 100, 3) (22498,)
print(x_test.shape, y_test.shape)   # (22499, 100, 100, 3) (22499,)

augment_size = 5000

randidx = np.random.randint(x_train.shape[0], size = augment_size) 
x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],      
    x_augmented.shape[1],     
    x_augmented.shape[2], 3)  

x_train = x_train.reshape(22498,100,100,3)
x_test = x_test.reshape(22499,100,100,3)

print(x_train.shape, x_test.shape)  # (40497, 100, 100, 3) (4500, 100, 100, 3)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)   

x_train = x_train.reshape(22498,100,100,3)
x_test = x_test.reshape(22499,100,100,3)

## numpy에서 데이터 합치기
# x_train = np.concatenate((x_train, x_augmented))
# y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)     # (22498, 100, 100, 3) (22498,)

print(np.unique(y_train, return_counts=True))

x_train = x_train.reshape(22498,100*100*3)
x_test = x_test.reshape(22499,100*100*3)

from sklearn.decomposition import PCA

pca = PCA(n_components=x_train.shape[0])  
# pca = PCA(n_components = None, svd_solver = 'full')

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 1659  
# 0.99 이상 : 5694  
# 0.999 이상 : 10707
# 1.0 일 때 : 16439

from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for i in range(len(lr)): 

    #2. 모델 구성
    model = Sequential()
    model.add(Dense(1024, input_shape=(x_train.shape[1],)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras68/18_kaggle_cat_dog/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml05_', str(i+1), '_', date, '_', filename])   
    #####################################

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=0,     
        save_best_only=True,   
        filepath=filepath, 
    )

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=2048,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)

    y_pre = model.predict(x_test, verbose=0)

    r2 = r2_score(y_test, y_pre)  
    
    print('결과', i+1)
    print('lr :',lr[i])
    print('loss :', round(loss[0],8))
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")
    


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

[Conv1D]
loss : 0.5863834023475647
acc : 0.68444
걸린 시간 : 38.78 초
accuracy_score : 0.6844444444444444
"""

# 결과 1
# PCA : 1659
# loss : 0.57263792
# acc : 0.70558691
# r2 score : 0.22485605340979975
# 걸린 시간 : 4.32 초
# ===============================================
# 결과 2
# PCA : 5694
# loss : 0.5814088
# acc : 0.72034311
# r2 score : 0.2308668599907555
# 걸린 시간 : 6.77 초
# ===============================================
# 결과 3
# PCA : 10707
# loss : 0.58511138
# acc : 0.73341036
# r2 score : 0.24326606020625874
# 걸린 시간 : 9.18 초
# ===============================================
# 결과 4
# PCA : 16439
# loss : 0.60988945
# acc : 0.69078624
# r2 score : 0.17215000217754328
# 걸린 시간 : 24.1 초
# ===============================================



### learning rate ###
# 결과 1
# lr : 0.1
# loss : 0.69327164
# acc : 0.5004667
# r2 score : -0.000246651771038664
# 걸린 시간 : 19.04 초


