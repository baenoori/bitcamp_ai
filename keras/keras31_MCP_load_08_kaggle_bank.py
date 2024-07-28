# 30_8 copy
# mcp load
# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/kaggle/playground-series-s4e1/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())

# 문자열 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']


print(x.shape)  # (165034, 10)
print(y.shape)  # (165034,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5324)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(64, input_dim=10, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))



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

# path = 'C:/ai5/_save/keras30_mcp/08_kaggle_bank/'
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
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=64,
#           verbose=1, 
#           validation_split=0.1,
#           callbacks=[es, mcp],
#           )
# end = time.time()

#4. 평가, 예측
print("============================ MCP 출력 ==============================")
model2 = load_model('C:/ai5/_save/keras30_mcp/08_kaggle_bank/k30_0726_2010_0015-0.1016.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)

# print("걸린 시간 :", round(end-start,2),'초')


# ### csv 파일 ###
# y_submit = model.predict(test_csv)

# # print(y_submit)
# y_submit = np.round(y_submit)
# # print(y_submit)
# sampleSubmission_csv['Exited'] = y_submit
# # print(sampleSubmission_csv)
# sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

# print(sampleSubmission_csv['Exited'].value_counts())


"""
loss : 0.10003902018070221
r2 score : 0.3991722537593737
acc_score : 0.8615487154629181

loss : 0.10003902018070221
r2 score : 0.3991722537593737

"""
