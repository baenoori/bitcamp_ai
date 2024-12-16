# DNN -> CNN

import sklearn as sk
from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D,Flatten
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)  # (20640, 8)

x = x.reshape(20640,2,2,2)
x = x/255.


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5555)

#### scaling (데이터 전처리)
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(2,2,2), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k39_02_', date, '_', filename])    
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_predict = model.predict(x_test)
print("걸린 시간 :", round(end-start,2),'초')


"""
loss : 0.29207977652549744
r2 score :  0.7753246532346505

[drop out]
loss : 0.28231775760650635
r2 score :  0.7828337787471844

[함수형 모델]
loss : 0.5426760911941528
r2 score :  0.5825593270822416

[CPU]
loss : 0.31958243250846863
r2 score :  0.7541688811039294
걸린 시간 : 11.01 초
GPU 없다!~!

[GPU]
loss : 0.31930163502693176
r2 score :  0.7543848734863335
걸린 시간 : 126.02 초
GPU 돈다!~!

[DNN -> CNN ]
loss : 0.4736214876174927
acc : 0.00178
걸린 시간 : 86.85 초
"""

