from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)

#1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# one hot encoding
y = pd.get_dummies(y)
# print(y.shape)  # (581012, 7)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5353, stratify=y)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=x_train.shape[1])  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 43
# 0.99 이상 : 49
# 0.999 이상 : 51
# 1.0 일 때 : 52

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]    

for i in range(len(lr)): 
    #2. 모델
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4)) 
    model.add(Dense(7, activation='softmax'))

    #3. 컴파일, 훈련
    from tensorflow.keras.optimizers import Adam
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras68/10_fetch_convtype/'
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
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=1000,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)

    y_pre = model.predict(x_test)

    r2 = r2_score(y_test, y_pre)  
    
    print('결과', i+1)
    print('lr :', lr[i])
    print('loss :', round(loss[0],8))
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


"""
loss : 0.013322586193680763
r2 score : 0.8210787449521321
acc_score : 0.9393652542081168
걸린 시간 : 120.93 초

loss : 0.015547509305179119
r2 score : 0.7777253594179163
acc_score : 0.9300024095556091
걸린 시간 : 114.98 초

"""
### PCA ###
# 결과 1
# PCA : 10
# loss : 0.39624023
# acc : 0.8542735
# r2 score : 0.6141057991431959
# 걸린 시간 : 179.39 초
# ===============================================
# Restoring model weights from the end of the best epoch: 116.
# Epoch 00126: early stopping
# 결과 2
# PCA : 12
# loss : 0.33333418
# acc : 0.87346393
# r2 score : 0.6309844842285411
# 걸린 시간 : 229.33 초
# ===============================================
# Restoring model weights from the end of the best epoch: 113.
# Epoch 00123: early stopping
# 결과 3
# PCA : 13
# loss : 0.31856462
# acc : 0.884152
# r2 score : 0.6697704337880197
# 걸린 시간 : 235.76 초
# ===============================================



#### learning rate ####
# 결과 1
# lr : 0.1
# loss : 1.20530236
# acc : 0.48760799
# r2 score : -5.2117057633083164e-05
# 걸린 시간 : 34.92 초
# ===============================================
# Restoring model weights from the end of the best epoch: 7.
# Epoch 00017: early stopping
# 결과 2
# lr : 0.01
# loss : 0.42565382
# acc : 0.83021241
# r2 score : 0.4686944261159343
# 걸린 시간 : 28.42 초
# ===============================================
# Restoring model weights from the end of the best epoch: 28.
# Epoch 00038: early stopping
# 결과 3
# lr : 0.005
# loss : 0.27704588
# acc : 0.89122576
# r2 score : 0.6681450843585223
# 걸린 시간 : 67.22 초
# ===============================================
# Restoring model weights from the end of the best epoch: 111.
# Epoch 00121: early stopping
# 결과 4
# lr : 0.001
# loss : 0.16198139
# acc : 0.93671477
# r2 score : 0.8218066147181323
# 걸린 시간 : 218.21 초
# ===============================================
# Restoring model weights from the end of the best epoch: 134.
# Epoch 00144: early stopping
# 결과 5
# lr : 0.0005
# loss : 0.17063886
# acc : 0.93354791
# r2 score : 0.8197500671743325
# 걸린 시간 : 237.59 초
# ===============================================
# Restoring model weights from the end of the best epoch: 222.
# Epoch 00232: early stopping
# 결과 6
# lr : 0.0001
# loss : 0.21661484
# acc : 0.91798908
# r2 score : 0.7736950197705913
# 걸린 시간 : 391.64 초
# ===============================================

