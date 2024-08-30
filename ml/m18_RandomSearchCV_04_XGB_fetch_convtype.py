from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)

#1. 데이터 
x, y = fetch_covtype(return_X_y=True)


# one hot encoding
# y = pd.get_dummies(y)
# print(y.shape)  # (581012, 7)
# print(y)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5353)

####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=3333)

parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist'], 'learning_rate':[0.1,0.01,0.001,0.005]}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist'], 'learning_rate':[0.01,0.001,0.005]}, #4
] #48


#2. 모델
model = RandomizedSearchCV(xgb.XGBClassifier(), parameters, cv=kfold, 
                     verbose=1, 
                     refit=True,    # 가장 좋은 모델 한번 더 돌리기 
                     n_jobs=-1,     # cpu 모든 코어 다 쓰기 (24개의 코어) 
                    n_iter=11,
                    random_state=1235
                     )   

st = time.time()
model.fit(x_train, y_train,
        #   eval_set = [(x_train,y_train), (x_test,y_test)],    # validation data 설정
        #   verbose=True
          )
et = time.time()

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)
print('best_score :', model.best_score_)    # train만 들어간 score, 훈련에서의 최고점
print('model.score', model.score(x_test, y_test))   # test로 뽑은 score
y_pre = model.predict(x_test)
y_pre = le.inverse_transform(y_pre)
print('accuracy_score :', accuracy_score(y_test,y_pre))
y_pre_best = model.best_estimator_.predict(x_test)
y_pre_best = le.inverse_transform(y_pre_best)
print('최적 튠 ACC :', accuracy_score(y_test, y_pre_best))      # 요걸로 확인하기
print('걸린 시간 :', round(et-st, 2), '초')

# Grid
# model.score 0.009603800213417783
# accuracy_score : 0.9749578327768408
# 최적 튠 ACC : 0.9749578327768408
# 걸린 시간 : 2262.72 초

# Random
# model.score 0.023269422739320506
# accuracy_score : 0.9514302433651165
# 최적 튠 ACC : 0.9514302433651165
# 걸린 시간 : 417.58 초


# model.score 0.011841244707583215
# accuracy_score : 0.9722040549378679
# 최적 튠 ACC : 0.9722040549378679
# 걸린 시간 : 1523.24 초
