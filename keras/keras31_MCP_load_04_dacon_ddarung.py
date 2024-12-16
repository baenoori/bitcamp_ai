# 30_4 copy
# mcp load
# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd # csv 파일 땡겨오고 원하는 열, 행 가져오는데 쓰임
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score    #r2를 보조 지표로 사용
import time

#1. 데이터
path = "C:/ai5/_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # 점 하나(.) : 루트라는 뜻, index_col=0 : 0번째 열을 index로 취급해달라는 의미
print(train_csv)    # (id열 포함) [1459 rows x 11 columns] / (id열 제외) [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) 
print(test_csv)     # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)      
print(submission_csv)   # [715 rows x 1 columns], NaN : 결측치 (비어있는 데이터)

print(train_csv.shape)  # (1459, 10)
print(test_csv.shape)  # (715, 9)
print(submission_csv.shape)  # (715, 1)

print(train_csv.columns)    # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                            #       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                            #       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                            #       dtype='object')

print(train_csv.info())

############# 결측치 처리 1. 삭제 #############
# print(train_csv.isnull().sum()) # 결측치의 개수 출력
print(train_csv.isna().sum()) # 위와 동일

train_csv = train_csv.dropna()  # null 값 drop (삭제) 한다는 의미 
print(train_csv.isna().sum())
print(train_csv)    # [1328 rows x 10 columns]
print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())

# 결측치 처리- 평균값 넣기
test_csv = test_csv.fillna(test_csv.mean()) # 컬럼별 평균값을 집어넣음 
print(test_csv.info())  # (715, 9)


# train_csv에서 x, y로 분할
x = train_csv.drop(['count'], axis=1)    # 행 또는 열 삭제 [count]라는 axis=1 열 (axis=0은 행)
print(x)    # [1328 rows x 9 columns]

y = train_csv['count']  # count 컬럼만 y에 넣음
print(y.shape)    # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=512)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=9))
# model.add(Dense(10, activation='relu'))    
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))


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

# path = 'C:/ai5/_save/keras30_mcp/04_dacon_ddarung/'
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
model2 = load_model('C:/ai5/_save/keras30_mcp/04_dacon_ddarung/k30_0726_1744_0136-2834.0022.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)



# y_submit = model.predict(test_csv)
# ############# submission.csv 만들기 // count컬럼에 값만 넣어주기 #############
# submission_csv['count'] = y_submit
# # print(submission_csv)
# # print(submission_csv.shape) # (715, 1)
# print('loss :', loss)

# submission_csv.to_csv(path + "submission_val_0725_1730_RS.csv")    #csv 만들기

"""
loss : 2080.247802734375
r2 score : 0.6768706042077424

loss : 2080.247802734375
r2 score : 0.6768706042077424
"""


