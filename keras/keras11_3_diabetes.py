from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score


#1. 데이터 
datesets = load_diabetes()
x = datesets.data
y = datesets.target

print(x, y) # (442, 10)
print(x.shape, y.shape) # (442,)

#[실습] 만들기
# R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=722)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)


""" 
test_size : 0.2
random_state : 722
epo : 500
batch_size : 2
loss : 2616.662109375
r2 score : 0.6206178868645644
"""
