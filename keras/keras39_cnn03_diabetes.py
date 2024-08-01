# DNN -> CNN

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import time

#1. 데이터 
datesets = load_diabetes()
x = datesets.data
y = datesets.target

print(x.shape)      # (442, 10)

x = x.reshape(442, 5, 2, 1)
x = x/255.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=555)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(5,2,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1))

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
print(date)
print(type(date))  
date = date.strftime("%m%d_%H%M")
print(date)    
print(type(date))  

path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k39_03_', date, '_', filename])  
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



print("걸린 시간 :", round(end-start,2),'초')

    
"""
loss : 3089.77392578125
r2 score : 0.510417644688258

[drop out]
loss : 3138.000732421875
r2 score : 0.502775937187783

[함수형 모델]
loss : 3113.980712890625
r2 score : 0.5065820332231512

[CPU]
loss : 3240.6484375
r2 score : 0.486511112726567
걸린 시간 : 2.37 초
GPU 없다!~!

[GPU]
loss : 3132.83837890625
r2 score : 0.5035939768293065
걸린 시간 : 4.65 초
GPU 돈다!~!

[DNN -> CNN]
loss : 6097.47607421875
acc : 0.0
걸린 시간 : 13.39 초
"""