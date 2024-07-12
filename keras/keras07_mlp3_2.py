import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]
            ])

print(x.shape)  #(3, 10)
print(y.shape)  #(3, 10)

x = x.T
y = y.T

print(x.shape)  #(10, 3)
print(y.shape)  #(10, 3)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim = 3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])

# np.set_printoptions(precision=3, suppress=True)   #소수점 precision 자리수까지 보기

print("loss :", loss)
print("[10,31,211]의 예측값 :", results)

""" 
결과
loss : 0.0017576201353222132
[10,31,211]의 예측값 :  [[11.029  0.013 -1.023]]
"""

