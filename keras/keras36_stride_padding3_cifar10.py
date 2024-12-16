import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')  # 데이터 그대로 가져옴
# plt.show()

##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

##### OHE
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

#2. 모델 구성 
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3), strides=1, )) 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu', strides=1))                     
model.add(Flatten())                            

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# path = './_save/keras35/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
# filepath = "".join([path, 'k35_06_', date, '_', filename])   
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

start = time.time()
hist = model.fit(x_train, y_train, epochs=2000, batch_size=64,
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
loss : 0.977311909198761
acc : 0.66
accuracy_score : 0.6608
걸린 시간 : 238.01 초

[stride, padding 'valid']
loss : 1.1081308126449585
acc : 0.62
accuracy_score : 0.6186
걸린 시간 : 57.98 초
"""




