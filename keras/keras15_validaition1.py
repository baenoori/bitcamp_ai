#14 (08-1) 카피
#validation

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6]) 
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])


#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_data=(x_val, y_val)    # << 추가됨
          )

# varbose = 0 : 침묵
# varbose = 1 : 디폴트
# varbose = 2 : 프로그래스바 삭제
# varbose = 나머지 : 에포만 나옴

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++") # 해당 열 아래의 평가 test의 평가 결과, 위는 train 데이터의 학습 결과, 아래 데이터로 판단 
loss = model.evaluate(x_test, y_test)
results = model.predict([11])

print("loss :", loss)
print("[11]의 예측값 :", results)

