# 09_2 copy
# 검색 R2

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # mse 기준, train 데이터로, 더 좋은 loss값
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # mse 기준, test 데이터로
print("loss :", loss)

#sklearn 라이브러리를 이용한 R2 구하기
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)    # loss를 테스트한 데이터와 동일한 걸로 평가해야함 

print("r2 스코어 :", r2)    # 하이퍼 파라미터 튜닝을 통해 r2스코어를 높임

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)    
print('RMSE :', rmse)

"""  
결과 예시
loss : 12.05027961730957
r2 스코어 : 0.7343376445919774
"""

"""
loss : 11.791112899780273
r2 스코어 : 0.7400512116068101
RMSE : 3.4338192400769074
"""