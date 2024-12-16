#35_3 copy

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import time
from sklearn.metrics import r2_score, accuracy_score

# #1. 데이터
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)  

# # x reshape -> (60000, 28, 28, 1)
# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

# print(x_train.shape, y_train.shape) 
# print(x_test.shape, y_test.shape) 

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

#2. 모델 구성 
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),
                 strides=1, 
                 padding='same',
                # padding='valid'     # 디폴트, padding적용 x
                 ))
model.add(Conv2D(filters=9, kernel_size=(3,3),
                 strides=1,
                 padding='valid'
                 ))   
model.add(Conv2D(8, (4,4)))                       

# model.add(Flatten())
# model.add(Dense(units=8))
# model.add(Dense(units=9, input_shape=(8,)))
#                             # shape = (batch_size, input_dim)
# model.add(Dense(units=10, activation='softmax'))
model.summary()

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10, verbose=1,
#                    restore_best_weights=True,
#                    )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")


# path = './_save/keras36/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
# filepath = "".join([path, 'k35_', date, '_', filename])   
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1, batch_size=128,
#           verbose=1, 
#           validation_split=0.1,
#           callbacks=[es, mcp],
#           )
# end = time.time()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)
# print('loss :', loss[0])
# print('acc :', round(loss[1],2))

# y_pre = model.predict(x_test)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)

# print("걸린 시간 :", round(end-start,2),'초')



