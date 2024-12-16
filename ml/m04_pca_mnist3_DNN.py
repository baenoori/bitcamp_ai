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

### PCA  <- 비지도 학습 

for i in range(4): 
    num = [154, 331, 486, 713, 784]
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    # evr = pca.explained_variance_ratio_     # 설명가능한 변화율

    # evr_cumsum = np.cumsum(evr)     #누적합
    # print(evr_cumsum)

    # print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
    # print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
    # print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
    # print('1.0 일 때 :', np.argmax(evr_cumsum >= 1.0)+1)

    #2. 모델
    model = Sequential()
    model.add (Dense(128, input_shape=(num[i],)))
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/ml04/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml04_', str(i+1), '_', date, '_', filename])   
    #####################################

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=0,     
        save_best_only=True,   
        filepath=filepath, 
    )

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=5000, batch_size=128,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=1)

    y_pre = model.predict(x_test1)

    y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)
    
    r2 = accuracy_score(y_test1, y_pre)
  
    print('결과', i+1)
    print('PCA :',num[i])
    print('acc :', round(loss[1],8))
    print('accuracy_score :', r2)   
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


# 결과 1
# PCA : 154
# acc : 0.9716
# 걸린 시간 : 19.04 초

# 결과 2
# PCA : 331
# acc : 0.96469998
# 걸린 시간 : 20.03 초

# 결과 3
# PCA : 486
# acc : 0.96890002
# 걸린 시간 : 22.81 초

# 결과 4
# PCA : 713
# acc : 0.96469998
# 걸린 시간 : 19.5 초



