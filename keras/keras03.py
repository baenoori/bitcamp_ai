from tensorflow.keras.models import Sequential  #순차적 딥러닝 모델이 들어있음
from tensorflow.keras.layers import Dense  #Dense layer 
import numpy as np  #수치데이터와 연산을 위해

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

epochs = 3200   # 이후 epoch 값을 출력하기 위해 미리 설정, epoch 값을 여기서 수정

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)   # 위에서 설정한 epoch 값 대입

#4. 평가, 예측
loss = model.evaluate(x,y)
print("====================================")
print("epochs : ", epochs)    #epoch 값도 출력하기 위해 13, 17 추가 및 수정
print("loss : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)


"""
[결과값]
epoch = 1000
loss :  0.38014310598373413
6의 예측값 :  [[5.7198462]]

epoch = 5000
loss :  0.37999990582466125
6의 예측값 :  [[5.7000027]]

epoch = 10000
loss :  0.3799998462200165
6의 예측값 :  [[5.7]]

epoch = 12000
loss :  0.37999996542930603
6의 예측값 :  [[5.6999993]]

epochs :  900
loss :  0.3812929391860962
6의 예측값 :  [[5.759393]]

epochs :  3200
loss :  0.37999996542930603
6의 예측값 :  [[5.700001]]

"""