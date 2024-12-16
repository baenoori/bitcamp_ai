from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import time
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터 
datesets = load_diabetes()
x = datesets.data
y = datesets.target


print(x.shape)  # (442, 10)

from sklearn.decomposition import PCA

pca = PCA(n_components=10)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 8
# 0.99 이상 : 8
# 0.999 이상 : 9
# 1.0 일 때 : 10

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=555)

### scaling ###
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


num = [8,9,10]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=num[i]))
    model.add(Dropout(0.1)) 
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1)) 
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))


    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=5000, batch_size=128,
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


# 결과 1
# PCA : 8
# acc : 0.0
# r2 score : 0.4727066717468176
# 걸린 시간 : 5.23 초
# ===============================================
# 결과 2
# PCA : 9
# acc : 0.0
# r2 score : 0.4904706294223352
# 걸린 시간 : 8.0 초
# ===============================================
# 결과 3
# PCA : 10
# acc : 0.0
# r2 score : 0.5041904596903626
# 걸린 시간 : 4.42 초
# ===============================================

### PF
# 0.4129547107930365
