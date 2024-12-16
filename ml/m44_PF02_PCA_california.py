import sklearn as sk
from sklearn.datasets import fetch_california_housing
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


print(x.shape)  # (20640, 8)

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 1
# 0.99 이상 : 1
# 0.999 이상 : 1
# 1.0 일 때 : 8

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5555)

#### scaling (데이터 전처리)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


num = [1,8]

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

    # r2 = accuracy_score(y_test, y_pre)
    r2 = r2_score(y_test, y_pre)  
    print('결과', i+1)
    print('PCA :',num[i])
    print('acc :', round(loss[1],8))
    # print('accuracy_score :', r2)       
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


# 결과 1
# PCA : 1
# acc : 0.00306848
# r2 score : 0.016387357429137328
# 걸린 시간 : 8.04 초
# ===============================================
# 결과 2
# PCA : 8
# acc : 0.00306848
# r2 score : 0.7671179836312663
# 걸린 시간 : 33.44 초
# ===============================================



### PF
# 0.5832447213709593



