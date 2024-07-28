# 30_3 copy
# mcp load

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import time

#1. 데이터 
datesets = load_diabetes()
x = datesets.data
y = datesets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=555)

### scaling ###
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=10))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
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
# print(date)
# print(type(date))  
# date = date.strftime("%m%d_%H%M")
# print(date)    
# print(type(date))  

# path = 'C:/ai5/_save/keras30_mcp/03_diabetes/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   
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
model2 = load_model('C:/ai5/_save/keras30_mcp/03_diabetes/k30_0726_1740_0150-2665.3804.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)



"""
loss : 3089.77392578125
r2 score : 0.510417644688258

loss : 3089.77392578125
r2 score : 0.510417644688258

"""

