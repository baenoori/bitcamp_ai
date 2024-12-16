# 16_2 california copy
# time, plot 추가 

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk
from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) # (2040, 8) (20640, )

#[실습] 만들기
# R2 0.59 이상

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5555)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          verbose=3,                # 추가
          validation_split=0.1      # 추가
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

print("걸린 시간 :", round(end-start, 2),'초')

print("=================== hist ==================")
print(hist)

print("================ hist.history =============")
print(hist.history)

print("================ loss =============")
print(hist.history['loss'])
print("================ val_loss =============")
print(hist.history['val_loss'])
print("==================================================")

# 시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'     # 한글 깨짐 해결, 폰트 적용

plt.figure(figsize=(9,6))   # 9 x 6 사이즈 
plt.plot(hist.history['loss'],c='red', label='loss',)  # y값 만 넣으면 시간 순으로 그려줌 
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')
plt.legend(loc='upper right')   # 우측 상단 label 표시
plt.title('캘리포니아 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시
plt.show()


""" 
test_size : 0.2
random_state : 189
epo : 500
batch_size : 40
loss : 0.6045974493026733
r2 score :  0.5415096393393787

<val 추가>
validation_split=0.1
loss : 0.5055875182151794
r2 score :  0.6110887813737476

<걸린 시간 추가>
epochs=1000
batch_size=32
loss : 0.4750061333179474
r2 score :  0.6346127283618905
걸린 시간 : 117.34 초
"""

