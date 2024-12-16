import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score   
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기 

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


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

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)


n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) 
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 


"""
loss : 0.0322132483124733
acc_score : 0.9532163742690059
걸린 시간 : 1.25 초

[drop out]
loss : 0.031158795580267906
acc_score : 0.9590643274853801
걸린 시간 : 1.3 초


KFold
acc :  [0.96491228 0.97368421 0.97368421 0.99122807 0.99115044] 
평균 acc : 0.9789

StratifiedKFold
acc :  [0.99122807 0.97368421 0.97368421 0.98245614 0.96460177] 
평균 acc : 0.9771
"""