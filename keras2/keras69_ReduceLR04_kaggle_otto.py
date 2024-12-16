import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=755)

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
# 0.95 이상 : 62
# 0.99 이상 : 82
# 0.999 이상 : 91
# 1.0 일 때 : 93

from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]


for i in range(len(lr)): 
    #2. 모델 구성
    model = Sequential()
    model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(9, activation='softmax'))

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

    path = './_save/keras69/13_kaggle_otto/'
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
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=64,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp, rlr],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)

    y_pre = model.predict(x_test, verbose=0)

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
"""

# 결과 1
# PCA : 62
# loss : 0.5430302
# acc : 0.79864252
# r2 score : 0.6318831515799652
# 걸린 시간 : 165.92 초
# ===============================================
# Restoring model weights from the end of the best epoch: 34.
# Epoch 00044: early stopping
# 결과 2
# PCA : 82
# loss : 0.52860922
# acc : 0.80284423
# r2 score : 0.643789239977156
# 걸린 시간 : 206.31 초
# ===============================================
# Restoring model weights from the end of the best epoch: 30.
# Epoch 00040: early stopping
# 결과 3
# PCA : 91
# loss : 0.54955304
# acc : 0.79347122
# r2 score : 0.6315086329049304
# 걸린 시간 : 202.26 초
# ===============================================
# Restoring model weights from the end of the best epoch: 22.
# Epoch 00032: early stopping
# 결과 4
# PCA : 93
# loss : 0.54575497
# acc : 0.80090499
# r2 score : 0.6320868911185771
# 걸린 시간 : 161.08 초
# ===============================================



### learning rate ### 

# 결과 1
# lr : 0.1
# loss : 1.95404196
# acc : 0.26535231
# r2 score : -0.0011027185831053998
# 걸린 시간 : 199.18 초
# ===============================================
# 결과 2
# lr : 0.01
# loss : 1.03049135
# acc : 0.62992889
# r2 score : 0.2891144469617295
# loss : 1.03049135
# acc : 0.62992889
# r2 score : 0.2891144469617295
# 걸린 시간 : 145.34 초
# ===============================================
# 결과 3
# lr : 0.005
# loss : 0.90742171
# lr : 0.005
# loss : 0.90742171
# acc : 0.69457012
# r2 score : 0.39656362611954077
# 걸린 시간 : 151.45 초
# ===============================================
# 결과 4
# lr : 0.001
# loss : 0.53945714
# acc : 0.8083387
# r2 score : 0.6508040436029174
# 걸린 시간 : 431.45 초
# ===============================================
# 결과 5
# lr : 0.0005
# loss : 0.5289669
# acc : 0.80575305
# r2 score : 0.6466114595846502
# 걸린 시간 : 612.88 초
# ===============================================
# 결과 6
# lr : 0.0001
# loss : 0.49892631
# acc : 0.81932771
# r2 score : 0.6637181285835232
# 걸린 시간 : 593.06 초
# ===============================================


##################################### Reduce #############################################
# 결과 1
# lr : 0.1
# loss : 1.95152712
# acc : 0.26535231
# r2 score : -0.00043224427898458373
# 걸린 시간 : 164.82 초
# ===============================================
# 결과 2
# lr : 0.01
# loss : 1.40983701
# acc : 0.41305754
# r2 score : 0.11153061913056078
# 걸린 시간 : 46.77 초
# ===============================================
# 결과 3
# lr : 0.005
# loss : 0.94443673
# acc : 0.66160309
# r2 score : 0.36080753393733445
# 걸린 시간 : 52.31 초
# ===============================================
# 결과 4
# lr : 0.001
# loss : 0.55015546
# acc : 0.79993534
# r2 score : 0.6332938886220758
# 걸린 시간 : 115.59 초
# ===============================================
# 결과 5
# lr : 0.0005
# loss : 0.51505566
# acc : 0.81254041
# r2 score : 0.6547383594298346
# 걸린 시간 : 177.98 초
# ===============================================
# 결과 6
# lr : 0.0001
# loss : 0.49866134
# acc : 0.81609565
# r2 score : 0.6606806989052326
# 걸린 시간 : 305.19 초
# ===============================================
