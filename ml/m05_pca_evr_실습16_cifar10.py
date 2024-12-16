# CNN -> DNN

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')  # 데이터 그대로 가져옴
# plt.show()

##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

##### OHE
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

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

# 0.95 이상 : 217
# 0.99 이상 : 658
# 0.999 이상 : 1430
# 1.0 일 때 : 3072


num = [217, 658, 1430, 3072]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델 구성
    model = Sequential()
    model.add(Dense(512, input_shape=(num[i],)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

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

    path = './_save/ml05/16_cifar10/'
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
    hist = model.fit(x_train1, y_train, epochs=2000, batch_size=64,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    y_pre = model.predict(x_test1, verbose=0)

    r2 = r2_score(y_test, y_pre)  
    
    print('결과', i+1)
    print('PCA :',num[i])
    print('loss :', round(loss[0],8))
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


"""
loss : 0.977311909198761
acc : 0.66
accuracy_score : 0.6608
걸린 시간 : 238.01 초

[stride, padding 'valid']
loss : 1.1081308126449585
acc : 0.62
accuracy_score : 0.6186
걸린 시간 : 57.98 초

[Max Pooling]
loss : 0.9801340103149414
acc : 0.67
accuracy_score : 0.6701
걸린 시간 : 58.02 초

[[DNN]]
loss : 1.4645131826400757
acc : 0.49
accuracy_score : 0.4936
걸린 시간 : 56.65 초
"""


# 결과 1
# PCA : 217
# loss : 1.35870886
# acc : 0.52700001
# r2 score : 0.3231355406171336
# 걸린 시간 : 35.92 초
# ===============================================
# 결과 2
# PCA : 658
# loss : 1.39649522
# acc : 0.50800002
# r2 score : 0.3062509574797253
# 걸린 시간 : 38.98 초
# ===============================================
# 결과 3
# PCA : 1430
# loss : 1.46667981
# acc : 0.47929999
# r2 score : 0.2712475528674331
# 걸린 시간 : 31.25 초
# ===============================================
# 결과 4
# PCA : 3072
# loss : 1.48200393
# acc : 0.49340001
# r2 score : 0.2789824324383532
# 걸린 시간 : 37.87 초
# ===============================================



