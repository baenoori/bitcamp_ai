# 30_13 copy
# mcp load

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/kaggle/otto-group-product-classification-challenge/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_cosl=0)
sampleSubmission_cav = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])
# print(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape, y.shape)     # (61878, 93) (61878,)
 
# one hot encoder
y = pd.get_dummies(y)
print(y.shape)      # (61878, 9)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=755)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(128, input_dim=93, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(9, activation='softmax'))


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

# path = 'C:/ai5/_save/keras30_mcp/13_kaggle_otto/'
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
model2 = load_model('C:/ai5/_save/keras30_mcp/13_kaggle_otto/k30_0726_2037_0033-0.0322.hdf5')       
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score :', r2)


# ### csv 파일 만들기 ###
# y_submit = model.predict(test_csv)
# # print(y_submit)

# y_submit = np.round(y_submit,1)
# # print(y_submit)

# sampleSubmission_cav[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit

# sampleSubmission_cav.to_csv(path + "sampleSubmission_0725_1730_RS.csv")


"""
loss : 0.02978578582406044
r2 score : 0.6373460236010778

loss : 0.02978578582406044
r2 score : 0.6373460236010778

"""
