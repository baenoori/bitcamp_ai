import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=231)

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)


#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])    # acc와 mse 보조지표로 사용, 'acc'로 사용 가능
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 

start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',       
    mode = 'min',               
    patience = 10,              
    restore_best_weights=True, 
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es],       
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)   
print('loss :', loss[0])
print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   


y_pred = model.predict(x_test)
print(y_pred[:20])

# y_pred = round(y_pred)  # 0 or 1로 acc에 값을 넣기 위해 반올림
# print(y_pred)           # 오류 : y_pred 는 numpy인데 python 함수를 사용하려 해서 오류

y_pred = np.round(y_pred)  # numpy round 함수
print(y_pred[:20])

from sklearn.metrics import r2_score, accuracy_score    # sklearn 에서 acc 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')


"""
loss : 0.19949066638946533
r2 score : 0.17840658553441235
걸린 시간 : 3.16 초

y_predict - activation = sigmoid 
[[0.04882199]
 [0.9127074 ]
 [0.9498874 ]
 [0.92066765]
 [0.05514327]
 [0.9646554 ]
 [0.8752985 ]
 [0.9483111 ]
 [0.00156233]
 [0.96317804]]
 -> 0과 1 사이의 값으로 나옴

binary_crossentropy
loss : 0.18100936710834503
acc : 0.93

"""

