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
model.add(SimpleRNN(units=10, input_shape=(3,1))) # timesteps , features
model.add(Dense(7))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120
#  dense (Dense)               (None, 7)                 77
#  dense_1 (Dense)             (None, 1)                 8
# =================================================================
# Total params: 205
# Trainable params: 205
# Non-trainable params: 0
# _________________________________________________________________

# 파라미터 개수 = units * (units + bias + feature)
#              = units*units + units*bias + units*feature

