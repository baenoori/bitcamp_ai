import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train = x[:7] 
y_train = y[:7]

x_val = x[6:11]
y_val = y[6:11]

x_test = x[10:]
y_test = y[10:]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=100)

# print(x_train)
# print(x_test)
# print(x_val)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_data=(x_val, y_val)
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([17])
print('[17]의 예측값 :', result)


