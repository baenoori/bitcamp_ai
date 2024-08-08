import numpy as np

# a = np.array([[1,2,3,4,5,6,7,8,9,10], 
#               [9,8,7,6,5,4,3,2,1,0]]).reshape(10,2)
a = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [9,8,7,6,5,4,3,2,1,0]]).T 
# a = np.array([range(1,11),
#                 range(11,21)]).reshape(10,2)

print(a)  # (2, 10)

size = 5                    # timestep 사이즈

print(len(a))   # 10

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)    # (6, 5,)

x = bbb[:, :-1]
y = bbb[:, -1, 0]

print("----------------")
print(x)
print("----------------")
print(y)
print(x.shape, y.shape) # (6, 4, 2) (6, 2)


# a = np.array([range(1,11),
#                 range(11,21)])

# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset)):
#         for j in range(len(dataset[i])-size+1):
#             subset = dataset[i][j:j+size]
#             aaa.append(subset)
#     return(np.array(aaa))


# bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)    # (6, 5)

# x = bbb[:, :-1]
# y = bbb[:, -1]

# print(x, y)
# print(x.shape, y.shape) # (6, 4) (6,)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#2. 모델 구성
model = Sequential()
model.add(LSTM(units=32, input_shape=(4,2), return_sequences=True)) # timesteps , features
model.add(LSTM(32, return_sequences=True)) # timesteps , features
model.add(LSTM(32))
# Flaten 사용하는 방법도 있음 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=4, 
          verbose=3
          )

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)

y_pred = model.predict([[[7,3],[8,1],[9,1],[10,0]]])
print('11을 원함 결과 :', y_pred)

# loss : 0.00012746865104418248
# 11을 원함 결과 : [[10.487139]]

