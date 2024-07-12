import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7]) #train의 개수가 많을 수록 좋음, test 데이터는 안 쓰는 데이터(학습에 적용X)
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++") # 해당 열 아래의 평가 test의 평가 결과, 위는 train 데이터의 학습 결과, 아래 데이터로 판단 
loss = model.evaluate(x_test, y_test)
results = model.predict([11])

print("loss :", loss)
print("[11]의 예측값 :", results)
