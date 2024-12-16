# 30_5 copy
# mcp load

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  
import time

#1. 데이터
# path = "./_data/따릉이/"    # 상대경로 
path = 'C:/ai5/_data/kaggle/bike-sharing-demand/'   # 절대경로 , 파이썬에서 \\a는 '\a'로 취급 특수문자 쓸 때 주의
# path = 'C:/ai5/_data/bike-sharing-demand'       # 위와 동일, //도 가능 

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 'datatime'열은 인덱스 취급, 데이터로 X
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  # (10886, 11)
print(test_csv.shape)   # (6493, 8)
print(submission_csv.shape) # (6493, 1)
 
print(train_csv.columns)  
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')

print(train_csv.info()) # 결측치 없음 
print(test_csv.info())  # 결측치 없음

print(train_csv.describe()) # 합계, 평균, 표준편차, 최소, 최대, 중위값, 사분위값 등을 보여줌 [8 rows x 11 columns]

##### 결측치 확인 #####
print(train_csv.isna().sum())   # 0
print(train_csv.isnull().sum()) # 0

print(test_csv.isna().sum())    # 0
print(test_csv.isnull().sum())  # 0

#### x와 y 분리 #####
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [0, 0] < list (2개 이상은 리스트)
print(x)    # [10886 rows x 8 columns]
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=654)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=8))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1, activation='linear'))


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10, verbose=1,
#                    restore_best_weights=True,
#                    )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# print(date)    
# print(type(date))  
# date = date.strftime("%m%d_%H%M")
# print(date)     
# print(type(date))  

# path = 'C:/ai5/_save/keras30_mcp/05_kaggle_bike/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    
# filepath = "".join([path, 'k30_', date, '_', filename])    
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
#           verbose=1, 
#           validation_split=0.1,
#           callbacks=[es, mcp],
#           )
# end = time.time()


#4. 평가, 예측
print("============================ MCP 출력 ==============================")
model2 = load_model('C:/ai5/_save/keras30_mcp/05_kaggle_bike/k30_0726_1750_0020-21887.3047.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)

# print("걸린 시간 :", round(end-start),'초')
# # ### csv 파일 ###
# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# submission_csv.to_csv(path + "submission_0725_1720_RS.csv")


"""
loss : 21463.5078125
r2 : 0.3185081846417004

loss : 21463.5078125
r2 score : 0.3185081846417004
"""

