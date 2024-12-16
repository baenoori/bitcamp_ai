# CNN -> DNN

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

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

# 0.95 이상 : 187
# 0.99 이상 : 459
# 0.999 이상 : 674
# 1.0 일 때 : 784

num = [187, 459, 674, 784]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델 구성
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

    path = './_save/ml05/15_fashion/'
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
    hist = model.fit(x_train1, y_train, epochs=1000, batch_size=128,  
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
1epo
loss : 0.5463500618934631
acc : 0.81
accuracy_score : 0.8073
걸린 시간 : 5.73 초

loss : 0.2949977517127991
acc : 0.91
accuracy_score : 0.9076z
걸린 시간 : 49.41 초

[stide, padding]
loss : 0.2757048010826111
acc : 0.91
accuracy_score : 0.9092
걸린 시간 : 56.47 초

[Max Pooling]
loss : 0.25262877345085144
acc : 0.92
accuracy_score : 0.916
걸린 시간 : 59.99 초

[[DNN]]
loss : 0.34890422224998474
acc : 0.88
accuracy_score : 0.8767
걸린 시간 : 34.77 초
"""

# 결과 1
# PCA : 187
# loss : 0.3340188
# acc : 0.88230002
# r2 score : 0.8117678810444635
# 걸린 시간 : 22.38 초
# ===============================================
# 결과 2
# PCA : 459
# loss : 0.35384515
# acc : 0.87819999
# r2 score : 0.8047088893843476
# 걸린 시간 : 21.65 초
# ===============================================
# 결과 3
# PCA : 674
# loss : 0.38230294
# acc : 0.86479998
# r2 score : 0.7851948508408614
# 걸린 시간 : 16.21 초
# ===============================================
# 결과 4
# PCA : 784
# loss : 0.37979898
# acc : 0.86739999
# r2 score : 0.7891138447381406
# 걸린 시간 : 21.22 초
# ===============================================