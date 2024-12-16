import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score   
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기 

#1 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)           # 행과 열 개수 확인 
print(datasets.feature_names)   # 열 이름 

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>

# 0과 1의 개수가 몇개인지 찾아보기 
print(np.unique(y, return_counts=True))     # (array([0, 1]), array([212, 357], dtype=int64))

# print(y.value_count)                      # error
print(pd.DataFrame(y).value_counts())       # numpy 인 데이터를 pandas 의 dataframe 으로 바꿔줌
# 1    357
# 0    212
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=231)

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
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

# 0.95 이상 : 9
# 0.99 이상 : 15
# 0.999 이상 : 24
# 1.0 일 때 : 30


num = [9, 15, 24, 30]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(30, input_dim=num[i], activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.1)) 
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


"""
loss : 0.0322132483124733
acc_score : 0.9532163742690059
걸린 시간 : 1.25 초

[drop out]
loss : 0.031158795580267906
acc_score : 0.9590643274853801
걸린 시간 : 1.3 초
"""

# 결과 1
# PCA : 9
# acc : 0.9239766
# r2 score : 0.7230426602194341
# 걸린 시간 : 3.3 초
# ===============================================
# 결과 2
# PCA : 15
# acc : 0.94152045
# r2 score : 0.7930918867116656
# 걸린 시간 : 1.88 초
# ===============================================
# 결과 3
# PCA : 24
# acc : 0.94152045
# r2 score : 0.7699098401178417
# 걸린 시간 : 3.08 초
# ===============================================
# 결과 4
# PCA : 30
# acc : 0.94152045
# r2 score : 0.5704496996433872
# 걸린 시간 : 5.82 초
# ===============================================

### PF
# 0.97076023


    