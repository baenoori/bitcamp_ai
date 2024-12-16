# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y는 T (degC) 로 잡기, 자르는거는 마음대로 ~ (y :144개~)
# 맞추기 : 2016년 12월 31일 00시 10분부터 2017.01.01 00:00:00 까지 데이터 144개 (훈련에 쓰지 않음 )

import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
path1 = "C:/ai5/_data/kaggle/jena/"
datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv", index_col=0)

print(datasets.shape)   # (420551, 14)

y_cor = datasets[-144:]['T (degC)']            # 예측치 정답
print(y_cor.shape)    # (144,)

## 훈련할 데이터 자르기
x_data = datasets[:-288].drop(['T (degC)'], axis=1)
y_data = datasets[144:-144]['T (degC)']

print(x_data)       # [420407 rows x 13 columns]
print(y_data)       # Name: T (degC), Length: 420407, dtype: float64

size_x = 144 
size_y = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

# x = x[:-1]
# y = y[(size_x-size_y+1):]


print(x.shape, y.shape)     # (420120, 144, 13) (420120, 144)
# print(x, y)

# 예측을 위한 x 데이터 
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)
x_predict = x_predict.to_numpy()
print(x_predict)
print(x_predict.shape)  # (144, 13)
# x_predict = split_x(x_predict, size_x)
print(x_predict.shape)  # (144, 13)

x_predict = x_predict.reshape(1,144,13)

# print(y[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2532)


print(x_train.shape)        # (378108, 144, 13)
print(x_test.shape)         # (42012, 144, 13)

# ## 스케일링 추가 ###
from sklearn.preprocessing import StandardScaler
x_train = x_train.reshape(378108*144,13)
x_test = x_test.reshape(42012*144,13)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(378108,144,13)
x_test = x_test.reshape(42012,144,13)

x_predict = x_predict.reshape(144,13)
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(1,144,13)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#2. 모델 구성
model = Sequential()
model.add(LSTM(units=64, input_shape=(144, 13), return_sequences=True))
model.add(Dropout(0.2)) 
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))  
model.add(LSTM(64))
model.add(Flatten())
model.add(Dropout(0.5))  
model.add(Dense(64, activation='relu'))
model.add(Dense(144))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=3,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ###### 
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras55/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k55_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=3,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(x_train, y_train, epochs=2000, batch_size=512, 
          validation_split=0.1,
          callbacks=[es, mcp],
          verbose=3
          )
end = time.time()


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=512)
print('loss :', result)

y_pred = model.predict(x_predict, batch_size=512)
print(y_pred.shape)
print('시간 :', end-start)

print(y_pred)


# y_pred = np.round(y_pred,2)
# acc = accuracy_score(y_cor, y_pred)

# rmse 를 위해 shape 맞춰주기
y_pred = np.array([y_pred]).reshape(144,1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_cor, y_pred)    
print('RMSE :', rmse)

# loss : 16.146814346313477
# 시간 : 2048.653157234192
# RMSE : 1.785246707543831   k55_0809_1515_0131-5.7192



### SCV 파일 ###

# submit = pd.read_csv(path1 + "jena_climate_2009_2016.csv")

# submit = submit[['Date Time','T (degC)']]
# submit = submit.tail(144)
# print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

# submit['T (degC)'] = y_pred
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

# submit.to_csv(path1 + "jena_배누리.csv", index=False)
