# (N, 10, 1) -> (N,5,2) 로 하기 
# reshape

import numpy as np

a = np.array(range(1,101))
x_predict = np.array(range(96,106)) 

size = 11                    # timestep 사이즈

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)  

x = bbb[:, :-1]
y = bbb[:, -1]

print(x, y)
print(x.shape, y.shape) # (90, 10) (90,)

# x_predict = split_x(x_predict, 5)
print(x_predict)
print(x_predict.shape)      # (6, 5)

x = x.reshape(90,5,2)

print(x)
print(x.shape)  # (90, 5, 2)
x_predict = x_predict.reshape(1,5,2)
print(x_predict)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#2. 모델 구성
model = Sequential()
model.add(LSTM(units=32, input_shape=(5,2), return_sequences=True)) # timesteps , features
# model.add(LSTM(32, return_sequences=True)) # timesteps , features
model.add(LSTM(32))
# Flaten 사용하는 방법도 있음 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', 
                   patience=1000, verbose=3,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras54/'
filename = '{epoch:04d}-{loss:.4f}.hdf5' 
filepath = "".join([path, 'k54_4_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='loss',
    mode='auto',
    verbose=3,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(x, y, epochs=2000, batch_size=4, 
          validation_split=0.1,
          callbacks=[es, mcp],
          verbose=3
          )

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)

y_pred = model.predict(x_predict)
print('x_predict :', x_predict)
print('결과 :', y_pred)

# loss : 0.15724173188209534
# x_predict : [[[ 96  97]
#   [ 98  99]
#   [100 101]
#   [102 103]
#   [104 105]]]
# 결과 : [[101.60279]]


