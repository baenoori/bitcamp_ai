import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  


#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y ,train_size=0.65,random_state=100)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          verbose=1,
          validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)    # evaluate도 verbose 존재, 마지막 1/1
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

results = model.predict([18])
print('[18]의 예측값 :', results)


"""
loss : 4.926429815303723e-13
r2 score : 0.9999999999999801
[18]의 예측값 : [[18.000002]]
"""


