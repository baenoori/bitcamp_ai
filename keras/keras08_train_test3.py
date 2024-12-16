import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과  test를 섞어서 7:3으로 나눠라
# 힌트 : 사이킷런 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    # train_size= 0.7   # 생략 가능, 디폴트: 0.75
                                                    test_size = 0.3,    # 디폴트: 0.25
                                                    # shuffle = True    # 디폴트: True, 생략 가능
                                                    random_state = 21   # 랜덤 난수, 랜덤값 고정
                                                    )

print('x_train :', x_train)  # [ 4  6  7  1  5  9 10]
print('x_test :', x_test)   # [2 3 8]
print('y_train :', y_train)  # [ 4  6  7  1  5  9 10]
print('y_test :', y_test)   # [2 3 8]

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++") # 해당 열 아래의 평가 test의 평가 결과, 위는 train 데이터의 학습 결과, 아래 데이터로 판단 
loss = model.evaluate(x_test, y_test)
results = model.predict([11])

print("loss :", loss)
print("[11]의 예측값 :", results)

