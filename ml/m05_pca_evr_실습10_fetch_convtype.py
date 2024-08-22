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

num = [10, 12, 13]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(64, input_dim=num[i], activation='relu'))
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/ml05/10_fetch_convtype/'
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
    hist = model.fit(x_train1, y_train, epochs=1000, batch_size=1000,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    y_pre = model.predict(x_test1)

    r2 = r2_score(y_test, y_pre)  
    
    print('결과', i+1)
    print('PCA :',num[i])
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

