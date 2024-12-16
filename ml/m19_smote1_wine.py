import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
tf.random.set_seed(33) # seed 고정 (첫 가중치가 고정)


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (178, 13) (178,)
print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

### 2 데이터 삭제 
x = x[:-39]
y = y[:-39]
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.75, shuffle=True,
                                                    stratify=y
                                                    )

"""
#2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])    # onehot 하지 않아도 돌아감
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss :', results[0])
print('acc :', results[1])

y_pre = model.predict(x_test)
# print(y_pre)    # 3개의 값이 나오니 argmax 필요

y_pre = np.argmax(y_pre, axis=1)
# y_test = np.argmax(np.array(y_test), axis=1)
print(y_pre)    # [1 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 2 1 1 0 1 0 1 0] 

acc = accuracy_score(y_test, y_pre)
f1 = f1_score(y_test, y_pre, average='macro')

print('acc :', acc)
print('f1_score :', f1)
"""

# acc : 0.8571428571428571
# f1_score : 0.5970961887477314


# exit()
####################### SMOTE 적용 #############################
# pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('sklearn :', sk.__version__)   # sklearn : 1.5.1

print(np.unique(y_train, return_counts=True))
# 증폭 전 : (array([0, 1, 2]), array([44, 53,  6], dtype=int64))

smote = SMOTE(random_state=7777)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))
# 증폭 후 : (array([0, 1, 2]), array([53, 53, 53], dtype=int64))

###############################################################


#2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])    # onehot 하지 않아도 돌아감
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss :', results[0])
print('acc :', results[1])

y_pre = model.predict(x_test)
# print(y_pre)    # 3개의 값이 나오니 argmax 필요

y_pre = np.argmax(y_pre, axis=1)
# y_test = np.argmax(np.array(y_test), axis=1)
print(y_pre)    # [1 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 2 1 1 0 1 0 1 0] 

acc = accuracy_score(y_test, y_pre)
f1 = f1_score(y_test, y_pre, average='macro')

print('acc :', acc)
print('f1_score :', f1)

#### smote 전 ####
# acc : 0.8571428571428571
# f1_score : 0.5970961887477314

#### smote 후 ####
# acc : 0.8857142857142857
# f1_score : 0.6259259259259259




