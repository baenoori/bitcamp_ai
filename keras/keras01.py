import tensorflow as tf
print(tf.__version__)  # 2.7.4 (tensorflow 버전 출력)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()        # 순차적으로 연산하는 모델 
model.add(Dense(1, input_dim=1))  # 인풋 하나 - x , 아웃풋 하나 - y

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')   # 컴퓨터가 알아먹게 컴파일 
model.fit(x, y, epochs=100)   #fit : 훈련을 해라, x,y데이터로 훈련, epochs : n번 훈련

#4. 평가, 예측
result = model.predict(np.array([4]))    # 우리의 목적인 4를 예측해서 result에 저장
print("4의 예측값 : ", result)    # 실행 시 터미널에서 나온 loss : 오차값
