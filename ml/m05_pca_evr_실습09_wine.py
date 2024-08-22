# 30_9 copy
# mcp save 
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)      # (178, 13) (178,)
print(np.unique(y, return_counts=True))    # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

### one hot encoding ###
y = pd.get_dummies(y)
print(y)
print(y.shape)      # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=512,
                                                    stratify=y)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
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
# 0.95 이상 : 10
# 0.99 이상 : 12
# 0.999 이상 : 13
# 1.0 일 때 : 13

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
    model.add(Dropout(0.1)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4)) 
    model.add(Dense(3, activation='softmax'))

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

    path = './_save/ml05/09_wine/'
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
    hist = model.fit(x_train1, y_train, epochs=1000, batch_size=16,  
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
loss : 0.007697440683841705
r2 score : 0.9646814509235723
acc_score : 1.0

[drop out]
loss : 0.0003021926968358457
r2 score : 0.9986181057598148
acc_score : 1.0
"""

# 결과 1
# PCA : 10
# acc : 0.94444442
# r2 score : 0.8323069757084274
# 걸린 시간 : 5.48 초
# ===============================================
# Restoring model weights from the end of the best epoch: 78.
# Epoch 00088: early stopping
# 결과 2
# PCA : 12
# acc : 0.94444442
# r2 score : 0.8297754939904594
# 걸린 시간 : 10.3 초
# ===============================================
# Restoring model weights from the end of the best epoch: 60.
# Epoch 00070: early stopping
# 결과 3
# PCA : 13
# acc : 0.94444442
# r2 score : 0.8297718211970025
# 걸린 시간 : 8.1 초
# ===============================================