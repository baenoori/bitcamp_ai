# 30_1 copy
# mcp load

import sklearn as sk
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터 
dataset = load_boston()

x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(32, input_shape=(13,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10, verbose=1,
#                    restore_best_weights=True,
#                    )
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,      # 제일 좋은 값의 위치 확인을 위해
#     save_best_only=True,    # 가장 좋은 값 저장
#     filepath='./_save/keras29_mcp/keras29_mcp3.hdf5',  # 저장위치, h5나 hdf5 상관없음 
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
#           verbose=1, 
#           validation_split=0.1,
#           callbacks=[es, mcp],
#           )
# end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')

print("============================ MCP 출력 ==============================")
model2 = load_model('C:/ai5/_save/keras30_mcp/01_boston/k30_0726_1718_0073-18.2584.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)


"""
기존
loss : 12.40392780303955
r2 score : 0.8340526044825189
걸린 시간 : 2.39 초

MCP 파일
loss : 12.40392780303955
r2 score : 0.8340526044825189

"""