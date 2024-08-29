# DNN -> CNN
# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional, Conv1D
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.svm import SVC



#1. 데이터
path = "C:/ai5/_data/kaggle/playground-series-s4e1/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())

# 문자열 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']


print(x.shape)  # (165034, 10)
print(y.shape)  # (165034,)

x = x.to_numpy()
x = x/255.


n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=64,
    use_label_encoder=False,
    eval_metric='mlogloss',
)

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)     # fit 제공됨
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 

# KFold
# acc :  [0.79013543 0.78834793 0.78928712 0.7850153  0.78922014] 
# 평균 acc : 0.7884

# StratifiedKFold
# acc :  [0.86472566 0.86896719 0.86460448 0.86684643 0.86350966] 
# 평균 acc : 0.8657



# # r2 = r2_score(y_test, y_pred)
# # print('r2 score :', r2)
# print(y_pred)
# y_pred = np.round(y_pred) 
# accuracy_score = accuracy_score(y_test, y_pred)0
# print('acc_score :', accuracy_score)
# # print("걸린 시간 :", round(end-start,2),'초')
# print("걸린 시간 :", round(end-start,2),'초')

# ### csv 파일 ###
# y_submit = model.predict(test_csv)

# # print(y_submit)
# y_submit = np.round(y_submit)
# # print(y_submit)
# sampleSubmission_csv['Exited'] = y_submit
# # print(sampleSubmission_csv)
# sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

# print(sampleSubmission_csv['Exited'].value_counts())

"""
loss : 0.10003902018070221
r2 score : 0.3991722537593737
acc_score : 0.8615487154629181

[drop out]
loss : 0.09984055161476135
r2 score : 0.40036401637592434
acc_score : 0.8619728550654386

[함수형 모델]
loss : 0.09969830513000488
r2 score : 0.4012183233479154
acc_score : 0.8611245758603975

[CPU]
loss : 0.09977076947689056
r2 score : 0.4007835079476919
acc_score : 0.8607610276296656
걸린 시간 : 49.8 초
GPU 없다!~!

[GPU]
loss : 0.10007652640342712
r2 score : 0.39894686488398223
acc_score : 0.8614881240911294
걸린 시간 : 267.69 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.36444875597953796
acc : 0.84
acc_score : 0.8416141541444498
걸린 시간 : 191.57 초

[LSTM]
loss : 0.43411722779273987
acc : 0.81
acc_score : 0.8070164808531265
걸린 시간 : 136.92 초

[Conv1D]
loss : 0.4146052598953247
acc : 0.81
acc_score : 0.8124697043141057
걸린 시간 : 186.8 초
"""
