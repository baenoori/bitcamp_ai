#36_0 copy

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#2. 모델 구성 
model = Sequential()
# model.add(Conv2D(10, 3, input_shape=(28,28,1),      # (3,3) -> 3으로도 표기 가능 
model.add(Conv2D(10, (3,3), input_shape=(28,28,1),      # 26,26,10
                 strides=1, 
                # padding='same',
                # padding='valid' 
                 ))
# model.add(MaxPooling2D(pool_size=3, padding='same'))       # 13, 13, 10, poolsize 디폴트 2
model.add(MaxPooling2D())       # 13, 13, 10
model.add(Conv2D(filters=9, kernel_size=(3,3),       # 11, 11, 9
                 strides=1,
                 padding='valid'
                 ))   
model.add(Conv2D(8, (2,2)))                       # 10, 10, 8

# model.add(Flatten())
# model.add(Dense(units=8))
# model.add(Dense(units=9, input_shape=(8,)))
#                             # shape = (batch_size, input_dim)
# model.add(Dense(units=10, activation='softmax'))
model.summary()




