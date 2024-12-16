from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터 
x, y = load_digits(return_X_y=True)     # sklearn에서 데이터를 x,y 로 바로 반환

print(pd.value_counts(y, sort=False))   # 0~9 순서대로 정렬

y_ohe = pd.get_dummies(y)
print(y_ohe.shape)          # (1797, 10)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.1, random_state=7777,
                                                    stratify=y)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=x_train.shape[0])  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 25
# 0.99 이상 : 42
# 0.999 이상 : 54
# 1.0 일 때 : 61


num = [25, 42, 54, 61]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(64, input_dim=num[i], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )
    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=1000, batch_size=16,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es],
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
loss : 0.005590484477579594
r2 score : 0.9378835045441063
acc_score : 0.9555555555555556

[drop out]
loss : 0.005380750633776188
r2 score : 0.9402138763769592
acc_score : 0.9666666666666667

"""

# 결과 1
# PCA : 25
# loss : 0.31685463
# acc : 0.93333334
# r2 score : 0.8873100473772245
# 걸린 시간 : 7.5 초
# ===============================================
# Restoring model weights from the end of the best epoch: 22.
# Epoch 00032: early stopping
# 결과 2
# PCA : 42
# loss : 0.15630405
# acc : 0.97222221
# r2 score : 0.9478465650012234
# 걸린 시간 : 13.34 초
# ===============================================
# Restoring model weights from the end of the best epoch: 22.
# Epoch 00032: early stopping
# 결과 3
# PCA : 54
# loss : 0.13678174
# acc : 0.96666664
# r2 score : 0.9406806785337528
# 걸린 시간 : 13.64 초
# ===============================================
# Restoring model weights from the end of the best epoch: 20.
# Epoch 00030: early stopping
# 결과 4
# PCA : 61
# loss : 0.09746253
# acc : 0.97777778
# r2 score : 0.9543518380238589
# 걸린 시간 : 12.86 초
# ===============================================



### PF
# 0.87222224



