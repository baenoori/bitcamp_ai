import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Bidirectional
import time
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN, Input, Conv1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터 
path1 = "C:/ai5/_data/중간고사데이터/"
naver_data = pd.read_csv(path1 + "NAVER 240816.csv", index_col=0, thousands = ',')
하이브_data = pd.read_csv(path1 + "하이브 240816.csv", index_col=0, thousands = ',')
성우하이텍_data = pd.read_csv(path1 + "성우하이텍 240816.csv", index_col=0, thousands = ',')

## 전일비 아이콘 삭제 ## 
naver_data = naver_data.drop(['전일비'],axis=1)
하이브_data = 하이브_data.drop(['전일비'], axis=1)
성우하이텍_data = 성우하이텍_data.drop(['전일비'], axis=1)

naver_data = naver_data.rename(columns={'Unnamed: 6':'전일비'})
하이브_data = 하이브_data.rename(columns={'Unnamed: 6':'전일비'})
성우하이텍_data = 성우하이텍_data.rename(columns={'Unnamed: 6':'전일비'})

## 시간 역순 정렬 
naver_data = naver_data.loc[::-1]
하이브_data = 하이브_data.loc[::-1]
성우하이텍_data = 성우하이텍_data.loc[::-1]

## 결측치 확인 및 채우기
print(naver_data.isna().sum())      # 거래량, 금액 결측치 있음 
print(하이브_data.isna().sum())

print(naver_data.info())

# # print(naver_data.info())
naver_data['거래량'] = naver_data['거래량'].fillna(naver_data['거래량'].mean())
naver_data['금액(백만)'] = naver_data['금액(백만)'].fillna(naver_data['금액(백만)'].mean())

# naver_data = naver_data.dropna()
print(naver_data.isna().sum())      # 거래량, 금액 결측치 있음 

print(naver_data.shape)    # (5390, 15)

## split 및 train test 데이터 나누기 ##
naver_datasets = naver_data[-948:]        # 하이브 데이터와 동일한 개수로 
하이브_datasets = 하이브_data
성우하이텍_datasets = 성우하이텍_data['종가'][-948:]

print(naver_datasets.shape)     # (948, 15)
print(하이브_datasets.shape)     # (948, 15)
print(성우하이텍_datasets.shape)    # (948,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

size = 11
naver_datasets = split_x(naver_datasets,size)
하이브_datasets = split_x(하이브_datasets,size)

# print(naver_datasets) 
print(하이브_datasets.shape)

# y = split_x(성우하이텍_datasets[11:], size)
y = 성우하이텍_datasets[11:] 

x1_datasets = naver_datasets[:-1]
x2_datasets = 하이브_datasets[:-1]

x1_pre = naver_datasets[-1:]
x2_pre = 하이브_datasets[-1:]

print(x1_datasets.shape, x2_datasets.shape) # (927, 11, 15) (927, 11, 15)
print(x1_pre.shape, x2_pre.shape)   # (11, 11, 15) (11, 11, 15)
print(y.shape)  # (927,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, test_size=0.1, random_state=321)

print(x1_datasets.shape, x2_datasets.shape) 
print(x1_pre.shape, x2_pre.shape)   # # (1, 11, 15) (1, 11, 15)

print(x1_train.shape)   # (843, 11, 15)
print(x1_test.shape)    # (94, 11, 15)

## 성우 하이텍 8.16 종가 추가
y[-1] = 7420

## 스케일링 추가
x1_train = x1_train.reshape(843*11,15)
x2_train = x2_train.reshape(843*11,15)
x1_test = x1_test.reshape(94*11,15)
x2_test = x2_test.reshape(94*11,15)
x1_pre = x1_pre.reshape(11,15)
x2_pre = x2_pre.reshape(11,15)
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x2_train = scaler.transform(x2_train)
x1_test = scaler.transform(x1_test)
x2_test = scaler.transform(x2_test)
x1_pre = scaler.transform(x1_pre)
x2_pre = scaler.transform(x2_pre)
x1_train = x1_train.reshape(843,11,15)
x2_train = x2_train.reshape(843,11,15)
x1_test = x1_test.reshape(94,11,15)
x2_test = x2_test.reshape(94,11,15)
x1_pre = x1_pre.reshape(1,11,15)
x2_pre = x2_pre.reshape(1,11,15)

#2-1. 모델
input1 = Input(shape=(11,15))
dense1 = LSTM(32, activation='relu', name='bit1', return_sequences=True)(input1)
dense2 = Conv1D(64, 3, activation='relu', name='bit2')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Conv1D(128, 3, activation='relu', name='bit3')(drop1)
max1 = MaxPool1D()(dense3)
fla1 = Flatten()(max1)
dense4 = Dense(64, activation='relu', name='bit4')(fla1)
output1 = Dense(32, activation='relu', name='bit5')(dense4)

#2-2. 모델
input11 = Input(shape=(11,15))
dense11 = LSTM(32, activation='relu', name='bit11', return_sequences=True)(input11)
dense21 = Conv1D(64, 3, activation='relu', name='bit21')(dense11)
drop2 = Dropout(0.2)(dense21)
dense31 = Conv1D(128, 3, activation='relu', name='bit31')(drop2)
max2 = MaxPool1D()(dense31)
fla2 = Flatten()(max2)
dense41 = Dense(64, activation='relu', name='bit41')(fla2)
output11 = Dense(32, activation='relu', name='bit51')(dense41)

#2-3. 모델 합치기
from keras.layers.merge import concatenate, Concatenate
merge1 = concatenate([output1, output11], name='mg1')

merge2 = Dense(16, name='mg2', activation='relu')(merge1)
merge3 = Dense(8, name='mg3', activation='relu')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input11], outputs=last_output)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=50, verbose=3,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ###### 
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/중간고사가중치/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, '중간고사_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=3,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit([x1_train,x2_train], y_train, epochs=5000, batch_size=16, 
          validation_split=0.1, 
          callbacks=[es, mcp],
          verbose=1,
          )
end = time.time()

#4. 평가, 예측
result = model.evaluate([x1_test,x2_test], y_test)
print('loss :', result)
print('걸린 시간 :', round(end-start,2), '초')

y_pred = model.predict([x1_pre,x2_pre])

print('8월 19일 월요일 종가의 예측 결과:', y_pred)

# 중간고사_0816_1203_0041-1086031.6250.hdf5
# loss : 1654454.625
# 걸린 시간 : 509.35
# 월요일 종가의 예측 결과: [[7900.915]]

# 중간고사_0816_1219_0072-212543.6406.hdf5
# loss : 199960.046875
# 걸린 시간 : 820.07
# 월요일 종가의 예측 결과: [[6882.0234]]

# Conv1D
# 중간고사_0816_1249_0048-592387.3750.hdf5
# loss : 702467.375
# 걸린 시간 : 136.86
# 8월 19일 월요일 종가의 예측 결과: [[7232.688]]

# Conv1D
# 중간고사_0816_1256_0017-707982.5625.hdf5
# loss : 751275.375
# 걸린 시간 : 72.41
# 8월 19일 월요일 종가의 예측 결과: [[7746.39]]

# Conv1D - Standerdscaler epo 1000- pat 10
# 중간고사_0816_1305_0047-153063.5000.hdf5
# loss : 168103.890625
# 걸린 시간 : 118.31
# 8월 19일 월요일 종가의 예측 결과: [[7536.487]]

# Conv1D - standerdscaler epo5000-pat30
# 중간고사_0816_1310_0083-100717.8594.hdf5
# loss : 123719.640625
# 걸린 시간 : 257.54
# 8월 19일 월요일 종가의 예측 결과: [[8039.0913]]

# 중간고사_0816_1411_0236-87423.6562.hdf5
# loss : 108230.9140625
# 걸린 시간 : 1282.81
# 8월 19일 월요일 종가의 예측 결과: [[7714.728]]

# Conv1D LSTM 
# 중간고사_0816_1434_0113-76139.1953.hdf5
# loss : 90807.8828125
# 걸린 시간 : 3450.58 초
# 8월 19일 월요일 종가의 예측 결과: [[7657.7554]]

# 중간고사_0817_1259_0256-63810.3477.hdf5
# loss : 52952.66015625
# 걸린 시간 : 600.96 초
# 8월 19일 월요일 종가의 예측 결과: [[7588.432]]

# 중간고사_0817_1341_0156-75324.6094_7509.hdf5
# loss : 92636.703125
# 걸린 시간 : 621.75 초

# 8월 19일 월요일 종가의 예측 결과: [[7509.644]]

# 중간고사_0817_1412_0221-91153.7812_7363
# loss : 81360.0390625
# 걸린 시간 : 850.93 초
# 8월 19일 월요일 종가의 예측 결과: [[7363.7935]]

# 중간고사_0817_1600_0120-68155.8516.hdf5
# loss : 99366.5078125
# 걸린 시간 : 512.67 초
# 8월 19일 월요일 종가의 예측 결과: [[7433.439]]

# 중간고사_0818_1511_0321-90455.8281_7500
# oss : 71988.9140625
# 걸린 시간 : 331.4 초
# 8월 19일 월요일 종가의 예측 결과: [[7500.4146]]

# 중간고사_0818_1555_0116-103459.7969_7355
# loss : 108354.5625
# 걸린 시간 : 312.64 초
# 8월 19일 월요일 종가의 예측 결과: [[7355.7046]]


