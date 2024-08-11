from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   # 이미지 땡겨오기
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
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



np_path = "C:/ai5/_data/image/me/"

me_test = np.load(np_path + 'keras46_me_arr_me.npy')



# [[0.]]

# print('확률: ',1-y_predict2)



np_path = "C:/ai5/_data/_save_npy/gender/2/"
x_train = np.load(np_path + 'keras45_07_x_train2.npy')
y_train = np.load(np_path + 'keras45_07_y_train2.npy')

x_train_woman = x_train[np.where(y_train > 0.0)]        # 0.0 보다 큰 y값이 있는 인덱스 추출
y_train_woman = y_train[np.where(y_train > 0.0)]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=921)
end1 = time.time()

# 데이터 걸린시간 : 1.43 초

print(x_train.shape, y_train.shape) # (24450, 100, 100, 3) (24450,)
print(x_test.shape, y_test.shape)   # (2717, 100, 100, 3) (2717,)
# 데이터 걸린시간 : 48.61 초

train_datagen =  ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    # horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=0.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

augment_size = 8000   

print(len(x_train_woman))      # 9489
# x_train_woman = x_train_woman.reshape()

randidx = np.random.randint(x_train_woman.shape[0], size = augment_size) 
print(randidx)              
print(np.min(randidx), np.max(randidx)) 

# print(x_train[0].shape) 

x_augmented = x_train_woman[randidx].copy() 
y_augmented = y_train_woman[randidx].copy()

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
).next()[0]

print(x_augmented.shape)   

x_train = x_train.reshape(24450, 100, 100, 3)
x_test = x_test.reshape(2717, 100, 100, 3)

print(x_train.shape, x_test.shape)  # (24450, 100, 100, 3) (2717, 100, 100, 3)

## numpy에서 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)         # (32450, 100, 100, 3) (32450,)

print(np.unique(y_train, return_counts=True))       # (array([0., 1.], dtype=float32), array([15885, 16565], dtype=int64)) <- 남자여자 데이터 불균형 맞춰짐 


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


start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.1,
          callbacks=[es],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1,
                      batch_size=16
                      )
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test,batch_size=16)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre1 = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre1)
print('accuracy_score :', r2)


y_predict2 = model.predict(me_test)
print(y_predict2)


