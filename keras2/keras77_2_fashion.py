from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import r2_score, accuracy_score

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

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

augment_size = 40000      # 6만개 -> 10만개로 만들기 위해  

randidx = np.random.randint(x_train.shape[0], size = augment_size) # 60000, size = 4000
# print(x_train.shape[0])     # 60000
print(randidx)                # [32458 55299 30575 ... 26476 26753 44762] <- 4만개
print(np.min(randidx), np.max(randidx)) # 0 59998

print(x_train[0].shape)     # (28, 28)

x_augmented = x_train[randidx].copy()  # 4만개의 데이터 copy (40000,28,28,1), copy로 새로운 메모리 할당, 서로 영향 X
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)    # (40000, 28, 28)
print(y_augmented.shape)    # (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],       # 4만
    x_augmented.shape[1],       # 28
    x_augmented.shape[2], 1)    # 28, 1

print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)    # (40000, 28, 28, 1), 변환된 데이터 4만개

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

## numpy에서 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)

#### 만들기 #### 

print(np.unique(y_train, return_counts=True))

##### OHE
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.fit_transform(y_test)


#2. 모델 구성 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dropout(0.7))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
print("걸린 시간 :", round(end-start,2),'초')



"""
[Max Pooling]
loss : 0.25262877345085144
acc : 0.92
accuracy_score : 0.916
걸린 시간 : 59.99 초

[데이터 증폭]
loss : 0.38384371995925903
acc : 0.87
accuracy_score : 0.8702
걸린 시간 : 98.95 초

[GAP]
# loss : 0.37627115845680237
# acc : 0.88
"""






