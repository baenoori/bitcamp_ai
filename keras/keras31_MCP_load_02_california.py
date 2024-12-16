# 30_2 copy
# mcp load

import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5555)

#### scaling (데이터 전처리)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=8))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10, verbose=1,
#                    restore_best_weights=True,
#                    )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# print(date)     # 2024-07-26 16:49:48.004109
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")
# print(date)     # 0726_1655
# print(type(date))   # <class 'str'>

# path = 'C:/ai5/_save/keras30_mcp/02_california/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf56 형태로 파일 명 생성, fit에서의 반환값을 가져옴
# filepath = "".join([path, 'k30_', date, '_', filename])    
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
#           verbose=1, 
#           validation_split=0.1,
#           callbacks=[es, mcp],
#           )
# end = time.time()


#4. 평가, 예측
print("============================ MCP 출력 ==============================")
model2 = load_model('C:/ai5/_save/keras30_mcp/02_california/k30_0726_1731_0099-0.3048.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)


"""
loss : 0.29207977652549744
r2 score :  0.7753246532346505

[mcp load model]
loss : 0.29207977652549744
r2 score : 0.7753246532346505

"""

