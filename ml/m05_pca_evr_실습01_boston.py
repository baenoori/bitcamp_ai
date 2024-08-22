import sklearn as sk
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터 
dataset = load_boston()

x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

print(x.shape)  # (506, 13)

# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)

pca = PCA(n_components=x.shape[1])  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 2
# 0.99 이상 : 3
# 0.999 이상 : 6
# 1.0 일 때 : 13

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

num = [2,3,6,13]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델
    model = Sequential()
    model.add(Dense(64, input_shape=(num[i],)))
    model.add(Dropout(0.3))     # 64개의 30% 를 제외하고 훈련. 상위 레이어에 종속적 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3)) 
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/ml05/01_boston'
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
    hist = model.fit(x_train1, y_train, epochs=5000, batch_size=128,
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp],
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


# loss : 10.009329795837402
# r2 score : 0.8660890417120943
# 걸린 시간 : 4.4 초

# loss : 9.515417098999023
# r2 score : 0.8726969059751946
# 걸린 시간 : 4.15 초

# 결과 1
# PCA : 2
# acc : 0.0
# r2 score : 0.26496197369680463
# 걸린 시간 : 1.9 초
# ===============================================
# 결과 2
# PCA : 3
# acc : 0.0
# r2 score : 0.2569738843802508
# 걸린 시간 : 1.12 초
# ===============================================
# 결과 3
# PCA : 6
# acc : 0.0
# r2 score : 0.49862292310071965
# 걸린 시간 : 1.36 초
# ===============================================
# 결과 4
# PCA : 13
# acc : 0.0
# r2 score : 0.8103806176664192
# 걸린 시간 : 2.33 초
# ===============================================
