#12 dacon_ddarung copy
# verbose, validation 추가 

# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd # csv 파일 땡겨오고 원하는 열, 행 가져오는데 쓰임
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score    #r2를 보조 지표로 사용

#1. 데이터
path = "./_data/따릉이/"

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

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=9))
model.add(Dense(10, activation='relu'))    
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=8,
          verbose=0,            # 추가
          validation_split=0.3  # 추가
          ) 

#4. 평가, 예측
loss = model.evaluate(x_test,y_test,verbose=0)  # 추가
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

""" 
test_size : 0.2
random_state : 512
epo : 500
batch_size : 10
loss : 2583.295654296875
r2 score : 0.5863459725939788

test_size : 0.1
random_state : 2321
epo : 500
batch_size : 5
loss : 2333.740478515625
r2 score : 0.5614321483888584

test_size : 0.1
random_state : 522
epo : 500
batch_size : 5
loss : 2344.593017578125

test_size : 0.1
random_state : 522
epo : 500
batch_size : 3
loss : 2344.593017578125

test_size : 0.1
random_state : 512
epo : 1000
batch_size : 8
loss : 2284.923095703125
"""

# # submission 에 test_csv 예측값 넣기
# y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)   # (715, 1)


# ############# submission.csv 만들기 // count컬럼에 값만 넣어주기 #############
# submission_csv['count'] = y_submit
# print(submission_csv)
# print(submission_csv.shape) # (715, 1)
# print('loss :', loss)

# submission_csv.to_csv(path + "submission_val_0716.csv")    #csv 만들기


# # 파일 이름 수정하고 loss 값을 줄여서 dacon에 제출!




"""
[val 추가]
validation_split=0.3
loss : 2453.90185546875
r2 score : 0.6188301440327537
"""