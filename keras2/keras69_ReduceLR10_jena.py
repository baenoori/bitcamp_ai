# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y는 T (degC) 로 잡기, 자르는거는 마음대로 ~ (y :144개~)
# 맞추기 : 2016년 12월 31일 00시 10분부터 2017.01.01 00:00:00 까지 데이터 144개 (훈련에 쓰지 않음 )

import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN, Bidirectional, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
path1 = "C:/ai5/_data/kaggle/jena/"
datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv", index_col=0)

print(datasets.shape)   # (420551, 14)

y_cor = datasets[-144:]['T (degC)']            # 예측치 정답
print(y_cor.shape)    # (144,)

## 훈련할 데이터 자르기
x_data = datasets[:-288].drop(['T (degC)'], axis=1)
y_data = datasets[144:-144]['T (degC)']

print(x_data)       # [420407 rows x 13 columns]
print(y_data)       # Name: T (degC), Length: 420407, dtype: float64

size_x = 144 
size_y = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

# x = x[:-1]
# y = y[(size_x-size_y+1):]


print(x.shape, y.shape)     # (420120, 144, 13) (420120, 144)
# print(x, y)

# 예측을 위한 x 데이터 
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)
x_predict = x_predict.to_numpy()
print(x_predict)
print(x_predict.shape)  # (144, 13)
# x_predict = split_x(x_predict, size_x)
print(x_predict.shape)  # (144, 13)

x_predict = x_predict.reshape(1,144,13)

# print(y[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2532)


print(x_train.shape)        # (378108, 144, 13)
print(x_test.shape)         # (42012, 144, 13)

# ## 스케일링 추가 ###
from sklearn.preprocessing import StandardScaler
x_train = x_train.reshape(378108,144*13)
x_test = x_test.reshape(42012,144*13)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_predict = x_predict.reshape(144,13)
# x_predict = scaler.transform(x_predict)
# x_predict = x_predict.reshape(1,144*13)

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

# 0.95 이상 : 69
# 0.99 이상 : 155
# 0.999 이상 : 227
# 1.0 일 때 : 1872

from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]


for i in range(len(lr)): 
    #2. 모델 구성
    model = Sequential()
    model.add(Dense(256, input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(144))

    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr[i]), metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    rlr = ReduceLROnPlateau(monitor='val_loss',
                        mode='auto',
                        patience=5, verbose=1,
                        factor=0.8,                 # 50% 감축
                        )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/keras69/21_jena/'
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
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=512,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp, rlr],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)

    y_pre = model.predict(x_test, verbose=0)

    r2 = r2_score(y_test, y_pre)  
    
    print('결과', i+1)
    print('lr :',lr[i])
    print('loss :', round(loss[0],8))
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")


# # y_pred = model.predict(x_predict, batch_size=512)
# print(y_pred.shape)
# print('시간 :', end-start)

# print(y_pred)

# y_pred = np.round(y_pred,2)
# acc = accuracy_score(y_cor, y_pred)

# # rmse 를 위해 shape 맞춰주기 
# y_pred = np.array([y_pred]).reshape(144,1)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test,y_predict))
# rmse = RMSE(y_cor, y_pred)    
# print('RMSE :', rmse)


##### 기존 #####
# loss : 16.146814346313477
# 시간 : 2048.653157234192
# RMSE : 1.785246707543831   k55_0809_1515_0131-5.7192

#### bidirectional ####
# RMSE : 4.937940444308187

##### Conv1D #####
# loss : 5.8745808601379395
# 시간 : 723.930643081665
# RMSE : 1.9208189362390746

### SCV 파일 ###

# submit = pd.read_csv(path1 + "jena_climate_2009_2016.csv")

# submit = submit[['Date Time','T (degC)']]
# submit = submit.tail(144)
# print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

# submit['T (degC)'] = y_pred
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

# submit.to_csv(path1 + "jena_배누리.csv", index=False)


# 결과 1
# PCA : 69
# loss : 2.29562664
# acc : 0.07766829
# r2 score : 0.9675235307705982
# 걸린 시간 : 941.57 초
# ===============================================
# 결과 2
# PCA : 155
# loss : 2.0938344
# acc : 0.07940588
# r2 score : 0.9703758858270708
# 걸린 시간 : 774.11 초
# ===============================================
# 결과 3
# PCA : 227
# loss : 1.83118069
# acc : 0.08183376
# r2 score : 0.9740926039835962
# 걸린 시간 : 937.8 초
# ===============================================
# 결과 4
# PCA : 1872
# loss : 1.79920995
# acc : 0.09009331
# r2 score : 0.9745471057513004
# 걸린 시간 : 1016.17 초
# ===============================================


##################################### Reduce #############################################
# 결과 1
# lr : 0.1
# loss : 20.86137581
# acc : 0.00585547
# r2 score : 0.7048546134189932
# 걸린 시간 : 40.23 초
# ===============================================
# 결과 2
# lr : 0.01
# loss : 2.55619645
# acc : 0.07816814
# r2 score : 0.9638343522633394
# 걸린 시간 : 339.41 초
# ===============================================



