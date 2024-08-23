import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)
    
#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (200000, 200)
print(y.shape)  # (200000,)

print(pd.value_counts(y, sort=True))    # 이진 분류
# 0    179902
# 1     20098

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5233,
                                                    stratify=y)


####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

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

# 0.95 이상 : 186
# 0.99 이상 : 197
# 0.999 이상 : 200
# 1.0 일 때 : 200

from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]


for i in range(len(lr)): 
    #2. 모델
    model = Sequential()
    model.add(Dense(512, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['acc'])

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

    path = './_save/keras69/12_kaggle_santender/'
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
    print('lr :',lr[i])
    print('loss :', round(loss[0],8))
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


# ### csv 파일 만들기 ###
# y_submit = model.predict(test_csv)
# print(y_submit)

# y_submit = np.round(y_submit)
# print(y_submit)

# submission_csv['target'] = y_submit
# submission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

# print(submission_csv['target'].value_counts())


"""
loss : 0.10050000250339508
r2 score : -0.1117287381878822

[drop out]
loss : 0.10050000250339508
r2 score : -0.1117287381878822
"""
### PCA ###
# 결과 1
# PCA : 25
# loss : 0.28978258
# acc : 0.90079999
# r2 score : 0.09053189878665369
# 걸린 시간 : 16.77 초
# ===============================================
# Restoring model weights from the end of the best epoch: 9.
# Epoch 00019: early stopping
# 결과 2
# PCA : 42
# loss : 0.27379838
# acc : 0.90355003
# r2 score : 0.1407445183646795
# 걸린 시간 : 18.11 초
# ===============================================
# Restoring model weights from the end of the best epoch: 2.
# Epoch 00012: early stopping
# 결과 3
# PCA : 54
# loss : 0.27866608
# acc : 0.89950001
# r2 score : 0.12262643932331885
# 걸린 시간 : 11.55 초
# ===============================================
# Restoring model weights from the end of the best epoch: 9.
# Epoch 00019: early stopping
# 결과 4
# PCA : 61
# loss : 0.26389474
# acc : 0.90434998
# r2 score : 0.16562652823729584
# 걸린 시간 : 18.24 초
# ===============================================


# #### learning rate
# 결과 1
# lr : 0.1
# loss : 0.32618862
# acc : 0.89950001
# r2 score : -1.7511976675876895e-05
# 걸린 시간 : 16.87 초
# ===============================================
# Restoring model weights from the end of the best epoch: 13.
# Epoch 00023: early stopping
# 결과 2
# lr : 0.01
# loss : 0.25112271
# acc : 0.91000003
# r2 score : 0.2146471143690316
# 걸린 시간 : 22.75 초
# ===============================================
# Restoring model weights from the end of the best epoch: 2.
# Epoch 00012: early stopping
# 결과 3
# lr : 0.005
# loss : 0.23764487
# acc : 0.91299999
# r2 score : 0.2442764143673063
# 걸린 시간 : 12.25 초
# ===============================================
# Restoring model weights from the end of the best epoch: 10.
# Epoch 00020: early stopping
# 결과 4
# lr : 0.001
# loss : 0.24852803
# acc : 0.90864998
# r2 score : 0.21281721181820668
# 걸린 시간 : 15.25 초
# ===============================================
# Restoring model weights from the end of the best epoch: 9.
# Epoch 00019: early stopping
# 결과 5
# lr : 0.0005
# loss : 0.24058205
# acc : 0.91224998
# r2 score : 0.2367195030117456
# 걸린 시간 : 14.49 초
# ===============================================
# Restoring model weights from the end of the best epoch: 18.
# Epoch 00028: early stopping
# 결과 6
# lr : 0.0001
# loss : 0.23562855
# acc : 0.91420001
# r2 score : 0.2562996736065969
# 걸린 시간 : 21.55 초
# ===============================================


##################################### Reduce #############################################
# 결과 1
# lr : 0.1
# loss : 0.32616782
# acc : 0.89950001
# r2 score : 2.7836927395386013e-05
# 걸린 시간 : 15.39 초
# ===============================================
# 결과 2
# lr : 0.01
# loss : 0.2442227
# acc : 0.90969998
# r2 score : 0.22264059903117495
# 걸린 시간 : 28.53 초
# ===============================================
# 결과 3
# lr : 0.005
# loss : 0.23869817
# acc : 0.91229999
# r2 score : 0.24360085380240637
# 걸린 시간 : 13.13 초
# ===============================================
# 결과 4
# lr : 0.001
# loss : 0.24841695
# acc : 0.90995002
# r2 score : 0.21582373073604677
# 걸린 시간 : 19.64 초
# ===============================================
# 결과 5
# lr : 0.0005
# loss : 0.24155203
# acc : 0.91189998
# r2 score : 0.23576122231366525
# 걸린 시간 : 18.73 초
# ===============================================
# 결과 6
# lr : 0.0001
# loss : 0.23554067
# acc : 0.91404998
# r2 score : 0.2563695058905251
# 걸린 시간 : 27.26 초
# ===============================================
