# 기존 kaggle 데이터에서 
# 1. train_cav의 y를 casual과 registered로 잡는다.
#    그래서 훈련을 해서 test_cav의 casual과 registered를 predict 한다. 

# 2. test_csv에 casual과 registered 컬럼을 합친다 (파일을 만듦)

# 3. train_csv에 y를 count로 잡는다. 

# 4. 전체 훈련

# 5. test_csv 예측해서 submission에 붙인다. 


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  

#1. 데이터
# path = "./_data/따릉이/"    # 상대경로 
path = 'C:\\ai5\\_data\\bike-sharing-demand\\'   # 절대경로 , 파이썬에서 \\a는 '\a'로 취급 특수문자 쓸 때 주의

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 'datatime'열은 인덱스 취급, 데이터로 X
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)  # (10886, 11)
print(test_csv.shape)   # (6493, 8)
print(submission_csv.shape) # (6493, 1)
 
print(train_csv.columns)  

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
y = train_csv[['casual', 'registered']]
print(y.shape)  # (10886, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=439)

print(x_train.shape)    # (9797, 8)
print(x_test.shape)     # (1089, 8)
print(y_train.shape)    # (9797, 2)
print(y_test.shape)     # (1089, 2)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

### csv 파일 ###
y_submit = model.predict(test_csv)
print(test_csv.shape)   # (6493, 8)
print(y_submit.shape)   # (6493, 2)

print("test csv 타입 : ", type(test_csv))   # test csv 타입 :  <class 'pandas.core.frame.DataFrame'>
print("y_submit 타입 : ", type(y_submit))   # y_submit 타입 :  <class 'numpy.ndarray'>

test2_csv = test_csv    # 원래는 .copy 사용 해야함,,,
print(test2_csv.shape)  # (6493, 8)

test2_csv[['casual', 'registered']] = y_submit
print(test2_csv)        # [6493 rows x 10 columns]

test2_csv.to_csv(path + "test2.csv")

# submission_csv['count'] = y_submit
# submission_csv.to_csv(path + "submission_0717_1645.csv")


