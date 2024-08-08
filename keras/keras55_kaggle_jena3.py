# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y는 T (degC) 로 잡기, 자르는거는 마음대로 ~ (y :144개~)
# 맞추기 : 2016년 12월 31일 00시 10분부터 2017.01.01 00:00:00 까지 데이터 144개 (훈련에 쓰지 않음 )

import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
path1 = "C:/ai5/_data/kaggle/jena/"
datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv", index_col=0)

print(datasets.shape)   # (420551, 14)

a = datasets[:-144]
print(a.shape)      # (420407, 14)

y_cor = datasets[-144:]['T (degC)']            # 예측치 정답
print(y_cor.shape)    # (144,)


# """
x_data = datasets[:-144].drop(['T (degC)'], axis=1)
y_data = datasets[:-144]['T (degC)']

print(x_data)       # [420407 rows x 13 columns]
print(y_data)       # Name: T (degC), Length: 420407, dtype: float64

size_x = 288 
size_y = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

x = x[:-1]
y = y[size_x-size_y+1:]

x_predict = x[-1]

# print(x.shape)          # (419687, 720, 13)
print(y.shape)          # (419687, 144)
# print(x.shape, y.shape)     # (419687, 720, 13) (420263, 144)

# print(x, y)

# np_path = "c:/ai5/_data/_save_npy/kaggle_jena/"
# np.save(np_path + 'keras55_x_.npy', arr=x)
# np.save(np_path + 'keras55_y_.npy', arr=y)
# """


# np_path = "C:/ai5/_data/_save_npy/kaggle_jena/"
# x = np.load(np_path + 'keras55_x_.npy')
# y = np.load(np_path + 'keras55_y_.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1))) # timesteps , features
model.add(LSTM(units=16, input_shape=(720,13))) # timesteps , features
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras52/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k52_2_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(x_train, y_train, epochs=5, batch_size=400, 
          validation_split=0.1,
        #   callbacks=[es, mcp],
          verbose=3
          )

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=400)
print('loss :', result)


y_pred = model.predict(x_predict, batch_size=400)
print(y_pred)


acc = accuracy_score(y_cor, y_pred)
print(y_pred)

