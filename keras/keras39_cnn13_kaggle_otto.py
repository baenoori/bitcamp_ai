# DNN -> CNN

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/kaggle/otto-group-product-classification-challenge/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_cav = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.isna().sum())   # 0
# print(test_csv.isna().sum())    # 0

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])
# print(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# print(x.shape, y.shape)     # (61878, 93) (61878,)
 
# one hot encoder
y = pd.get_dummies(y)
# print(y.shape)      # (61878, 9)

x = x.to_numpy()
x = x.reshape(61878, 31, 3, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=755)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(31,3,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=9, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
# print(date)    
# print(type(date))  
date = date.strftime("%m%d_%H%M")
# print(date)     
# print(type(date))  

path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k39_13_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)
r2 = r2_score(y_test,y_pre)
print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if(gpus):
#     print('GPU 돈다!~!')
# else:
#     print('GPU 없다!~!')

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

[drop out]
loss : 0.029482578858733177
r2 score : 0.6487328463573875

[함수형 모델]
loss : 0.02975039929151535
r2 score : 0.6460698625631718

[CPU]
loss : 0.030084433034062386
r2 score : 0.6387725828611668
걸린 시간 : 203.31 초
GPU 없다!~!

[GPU]
loss : 0.029703930020332336
r2 score : 0.642266479275404
걸린 시간 : 178.28 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.5283812880516052
acc : 0.81
r2 score : 0.6421193170439987
걸린 시간 : 72.55 초

"""
