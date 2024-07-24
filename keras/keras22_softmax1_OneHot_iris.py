import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


#1.  데이터
datasets = load_iris()
# print(datasets)     # y가 0,1,2 <- 품종 3종류, x 4개의 컬럼
# print(datasets.DESCR)
# print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

print(y)        # 0, 1, 2가 순서대로 되어있음 suffle을 꼭 해주어야함 !!
print(np.unique(y, return_counts=True))     
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50



######## one hot encoding ########
y = pd.get_dummies(y)   # pd 이용
print(y)

# from sklearn.preprocessing import OneHotEncoder   # sklearn 이용
# y = y.reshape(-1,1)   # reshape 할 때 데이터의 값이 바뀌면 x, 순서가 바뀌면 x
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y)
# print(y)

# ohe = OneHotEncoder(sparse=False)   # True가 디폴트,
# ohe.fit(y)
# y = ohe.transform(y)      # scaler할 때 사용

# from tensorflow.keras.utils import to_categorical   # keras 이용
# y = to_categorical(y)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2321,
                                                    stratify=y)


#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))   # one-hot-encoding으로 y의 컬럼이 3으로 바뀜
# model.add(Dense(1, activation='linear'))


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])   # 다중분류에서의 loss 함수
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, batch_size=4,
                 verbose=3,
                 validation_split=0.2,
                 callbacks=[es]
                 )

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)
# print(y_pred)
y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')



"""
loss : 0.005562429782003164
acc : 1.0
r2 score : 0.9993626651204535
acc_score : 1.0
걸린 시간 : 7.11 초

stratify=y 추가

loss : 0.1818111538887024
acc : 0.933
r2 score : 0.8174484591437388
acc_score : 0.9333333333333333
걸린 시간 : 3.76 초
"""
