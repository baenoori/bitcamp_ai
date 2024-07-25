# 22_4 copy
# scaling 추가

from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터 
x, y = load_digits(return_X_y=True)     # sklearn에서 데이터를 x,y 로 바로 반환

print(x)
print(y)
print(x.shape, y.shape)     # (1797, 64) (1797,)

print(pd.value_counts(y, sort=False))   # 0~9 순서대로 정렬
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

y_ohe = pd.get_dummies(y)
print(y_ohe.shape)          # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.1, random_state=7777,
                                                    stratify=y)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=200,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=5000, batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],2))

y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end-start, 2), '초')


"""
epochs=5000, batch_size=1
random_state=7777
loss : 0.027419203892350197
acc : 0.99
r2 score : 0.989750870428856
acc_score : 0.9944444444444445
걸린 시간 : 177.92 초

[scaling 추가 - minmax]
loss : 0.15078093111515045
acc : 0.98
r2 score : 0.9603442204749065
acc_score : 0.9777777777777777
걸린 시간 : 159.58 초

[scaling - standard]
loss : 0.14644896984100342
acc : 0.98
r2 score : 0.9626700754530256
acc_score : 0.9833333333333333
걸린 시간 : 141.17 초

[scaling - MaxAbsScaler]
loss : 0.12007380276918411
acc : 0.96
r2 score : 0.9409348966976051
acc_score : 0.9611111111111111
걸린 시간 : 151.8 초

[scaling - RobustScaler]
loss : 0.1877421885728836
acc : 0.97
r2 score : 0.935301391408594
acc_score : 0.9555555555555556
걸린 시간 : 147.37 초

"""

