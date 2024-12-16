import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],           
             ])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)     # (7, 3) (7,)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)      # (7, 3, 1)
# 3-D tenser with shape (batch_size, timesteps, feature)

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1))) # timesteps , features
model.add(SimpleRNN(units=10, input_length=3, input_dim=1)) # (3, 1) = length, dim
# model.add(SimpleRNN(units=10, input_dim=1, input_length=3))     # 가독성 떨어짐, 쓰고싶으면 쓰기 
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)

x_pred = np.array([8,9,10]).reshape(1,3,1)      # [[[8],[9],[10]]]
y_pred = model.predict(x_pred)
# (3,) -> (1,3,1)

print('[8,9,10]의 결과 :', y_pred)
#[8,9,10]의 결과 : [[10.884811]]

# LSTM : [8,9,10]의 결과 : [[10.770827]]
# GRU : [8,9,10]의 결과 : [[10.621254]]



