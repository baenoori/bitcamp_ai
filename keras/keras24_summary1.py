from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #        파라미터 계산 시 바이어스도 1개의 노드로 계산
# =================================================================
#  dense (Dense)               (None, 3)                 6             <- (input(1)+bias(1)) * output(3) = 6

#  dense_1 (Dense)             (None, 4)                 16            <- (input(3)+bias(1)) * output(4) = 16

#  dense_2 (Dense)             (None, 3)                 15            <- (input(4)+bias(1)) * output(3) = 15

#  dense_3 (Dense)             (None, 1)                 4             <- (input(3)+bias(1)) * output(1) = 4

# Total params: 41              # 전체 연산량
# Trainable params: 41
# Non-trainable params: 0       # 가중치를 세이브해서 사용해 훈련시키지 않는 파라미터, 전위학습 



