import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0,1,1,0])

print(x_data.shape, y_data.shape)   # (4, 2) (4,)

#2. 모델 
# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=500)

#4. 평가, 예측
# acc = model.score(x_data, y_data)
# print('model.score : ', acc)
loss = model.evaluate(x_data, y_data)
print('acc :', loss[1])

y_pre = np.round(model.predict(x_data)).reshape(-1,).astype(int)
acc2 = accuracy_score(y_data, y_pre)
print('accuracy_score :', acc2)

print("---------------------")
print(y_data)
print(y_pre)

# LinearSVC 모델
# model.score :  0.5
# accuracy_score : 0.5
# ---------------------
# [0 1 1 0]
# [0 0 0 0]

# Perceptron 모델
# model.score :  0.5
# accuracy_score : 0.5
# ---------------------
# [0 1 1 0]
# [0 0 0 0]

# keras-Sequential 모델
# acc : 0.5
# accuracy_score : 0.5
# ---------------------
# [0 1 1 0]
# [0 1 0 1]

