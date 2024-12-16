import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱 => 7:3 으로 잘라라!

x_train = x[:7] # [:-3] 도 동일한 의미 
x_test = x[7:]  # [-3:] 도 동일한 의미

y_train = y[:7]
y_test = y[7:]

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++") # 해당 열 아래의 평가 test의 평가 결과, 위는 train 데이터의 학습 결과, 아래 데이터로 판단 
loss = model.evaluate(x_test, y_test)
results = model.predict([11])

print("loss :", loss)
print("[11]의 예측값 :", results)
