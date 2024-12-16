#13_1 copy
# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  
import time

#1. 데이터
# path = "./_data/따릉이/"    # 상대경로 
path = 'C:\\ai5\\_data\\bike-sharing-demand\\'   # 절대경로 , 파이썬에서 \\a는 '\a'로 취급 특수문자 쓸 때 주의
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
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True    
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

print("걸린 시간 :", round(end-start),'초')
# ### csv 파일 ###
# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# submission_csv.to_csv(path + "submission_0717_1645.csv")


print("=================== hist ==================")
print(hist)

print("================ hist.history =============")
print(hist.history)

print("================ loss =============")
print(hist.history['loss'])
print("================ val_loss =============")
print(hist.history['val_loss'])
print("==================================================")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'     # 한글 깨짐 해결, 폰트 적용

plt.figure(figsize=(9,6))   # 9 x 6 사이즈 
plt.plot(hist.history['loss'],c='red', label='loss',)  # y값 만 넣으면 시간 순으로 그려줌 
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')
plt.legend(loc='upper right')   # 우측 상단 label 표시
plt.title('캐글 바이크 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시
plt.show()


"""
epochs=100, batch_size=64
loss : 21989.11328125
r2 : 0.28736574424982986
걸린 시간 : 11 초


epochs=100, batch_size=64
loss : 21982.01171875
r2 : 0.3020451022166337
걸린 시간 : 5 초
"""



