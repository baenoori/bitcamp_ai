# 29_1 copy

import sklearn as sk
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터 
dataset = load_boston()
print(dataset)
print(dataset.DESCR) 
print(dataset.feature_names)


x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_shape=(13,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,      # 제일 좋은 값의 위치 확인을 위해
    save_best_only=True,    # 가장 좋은 값 저장
    filepath='./_save/keras29_mcp/keras29_mcp3.hdf5',  # 저장위치, h5나 hdf5 상관없음 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')


# loss : 10.009329795837402
# r2 score : 0.8660890417120943
# 걸린 시간 : 4.4 초


