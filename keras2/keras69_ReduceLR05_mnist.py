# m04_1에서 뽑은 4가지 결과로 4가지ㅁ 모델 만들기 
# input_shape()
# 1. 70000,154
# 2. 70000,332
# 3. 70000,544
# 4. 70000,683
# 5. 70000,713

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier  # 분류는 classifiaer, 회귀는 regress

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)

(x_train, y_train), (x_test, y_test) = mnist.load_data()   # y 데이터를 뽑지 않고 언더바 _ 로 자리만 남겨둠 
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# x = np.concatenate([x_train, x_test], axis=0)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

x_train = x_train/255.    
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

from tensorflow.keras.optimizers import Adam
lr = [0.01, 0.005, 0.001, 0.0005, 0.0001]

### PCA  <- 비지도 학습 

for i in range(len(lr)): 
    #2. 모델
    model = Sequential()
    model.add (Dense(128, input_shape=(x_train.shape[1],)))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(64, activation='relu'))
    model.add (Dense(64, activation='relu'))
    model.add (Dense(32, activation='relu'))
    model.add (Dense(32, activation='relu'))
    model.add (Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )
    rlr = ReduceLROnPlateau(monitor='val_loss',
                        mode='auto',
                        patience=5, verbose=1,
                        factor=0.8,                 # 50% 감축
                        )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras69/14_mnist/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml05_', str(i+1), '_', date, '_', filename])   
    #####################################

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=0,     
        save_best_only=True,   
        filepath=filepath, 
    )

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=5000, batch_size=128,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es, mcp, rlr],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)

    y_pre = model.predict(x_test)

    y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)
    
    r2 = accuracy_score(y_test1, y_pre)
  
    print('결과', i+1)
    print('lr :',lr[i])
    print('acc :', round(loss[1],8))
    print('accuracy_score :', r2)   
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


# 결과 1
# PCA : 154
# acc : 0.97170001
# accuracy_score : 0.9717
# 걸린 시간 : 20.81 초
# ===============================================
# 결과 2
# PCA : 331
# acc : 0.97189999
# accuracy_score : 0.9719
# 걸린 시간 : 30.77 초
# ===============================================
# 결과 3
# PCA : 486
# acc : 0.96799999
# accuracy_score : 0.968
# 걸린 시간 : 32.71 초
# ===============================================
# 결과 4
# PCA : 713
# acc : 0.97180003
# accuracy_score : 0.9718
# 걸린 시간 : 28.4 초
# ===============================================



### learning rate ### 
# 결과 1
# lr : 0.1
# acc : 0.1135
# accuracy_score : 0.1135
# 걸린 시간 : 80.33 초   
# ===============================================
# 313/313 [==============================] - 1s 2ms/step - loss: 0.3085 - acc: 0.9310
# 결과 2
# lr : 0.01
# acc : 0.93099999
# accuracy_score : 0.931
# 걸린 시간 : 41.72 초
# ===============================================
# 313/313 [==============================] - 1s 2ms/step - loss: 0.1239 - acc: 0.9696
# 결과 3
# lr : 0.005
# acc : 0.96960002
# accuracy_score : 0.9696
# 걸린 시간 : 73.13 초
# ===============================================
# 313/313 [==============================] - 1s 2ms/step - loss: 0.1030 - acc: 0.9742
# 결과 4
# lr : 0.001
# acc : 0.97420001
# accuracy_score : 0.9742
# 걸린 시간 : 95.7 초
# ===============================================
# 313/313 [==============================] - 2s 7ms/step - loss: 0.0965 - acc: 0.9730
# 결과 5
# lr : 0.0005
# acc : 0.97299999
# accuracy_score : 0.973
# 걸린 시간 : 71.46 초
# ===============================================
# 313/313 [==============================] - 1s 2ms/step - loss: 0.1075 - acc: 0.9699
# 결과 6
# lr : 0.0001
# acc : 0.96990001
# accuracy_score : 0.9699
# 걸린 시간 : 104.61 초
# ===============================================


##################################### Reduce #############################################
# 결과 1
# lr : 0.01
# acc : 0.93959999
# accuracy_score : 0.9396
# 걸린 시간 : 14.05 초
# ===============================================
# 결과 2
# lr : 0.005
# acc : 0.9702
# accuracy_score : 0.9702
# 걸린 시간 : 34.17 초
# ===============================================
# 결과 3
# lr : 0.001
# acc : 0.97610003
# accuracy_score : 0.9761
# 걸린 시간 : 29.07 초
# ===============================================
# 결과 4
# lr : 0.0005
# acc : 0.97359997
# accuracy_score : 0.9736
# 걸린 시간 : 27.09 초
# ===============================================
# 결과 5
# lr : 0.0001
# acc : 0.97130001
# accuracy_score : 0.9713
# 걸린 시간 : 40.38 초
# ===============================================

