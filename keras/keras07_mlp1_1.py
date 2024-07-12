import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([[1,2,3,4,5],
            #  [6,7,8,9,10]])  # 행렬 데이터, shape가 안맞음 (2, 5), 아래 코드가 맞는 shape
x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

print(x.shape)  # (5, 2)
print(y.shape)  # (5, )

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=2))  #2개의 열(컬럼, 벡터)가 있다는 의미 ★행 무시, 열 우선
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))  # y는 벡터 -> output 의 차원은 1

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[6, 11]])  # predict 에도 x데이터와 같은 형태의 데이터를 넣어주어야함, [[6]] 만 넣으면 오류
print("loss  :", loss)
print("[6, 11]의 예측값 :", results)

# [실습] : 소수 2째자리까지 맞추기

# predict에 [[6]] 입력 시 오류 
