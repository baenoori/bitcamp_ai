import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


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


num = [25, 42, 54, 61]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(512, input_dim=num[i], activation='relu'))
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/ml05/12_kaggle_santender/'
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

