# DNN -> CNN

from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional, Conv1D
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, r2_score


#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)      # (178, 13) (178,)
print(np.unique(y, return_counts=True))    # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

### one hot encoding ###
# y = pd.get_dummies(y)
print(y)
print(y.shape)      # (178, 3)

# x = x.reshape(178, 13, 1 )
# x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, 
                                                    stratify=y,
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# exit()
n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 

y_pre = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_pre)
print('cross_val_predict ACC :', acc)

# KFold
# acc :  [1.         0.97222222 1.         1.         0.97142857] 
# 평균 acc : 0.9887

# StratifiedKFold
# acc :  [1.         0.97222222 0.97222222 1.         0.94285714] 
# 평균 acc : 0.9775

# acc :  [0.93103448 0.96551724 1.         1.         1.        ] 
# 평균 acc : 0.9793
# cross_val_predict ACC : 1.0







# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)
# print('loss :', loss[0])
# print('acc :', round(loss[1],2))

# y_pred = model.predict(x_test)

# r2 = r2_score(y_test, y_pred)
# print('r2 score :', r2)
# y_pred = np.round(y_pred) 
# accuracy_score = accuracy_score(y_test, y_pred)
# print('acc_score :', accuracy_score)
# # print("걸린 시간 :", round(end-start,2),'초')

# print("걸린 시간 :", round(end-start,2),'초')

"""
loss : 0.007697440683841705
r2 score : 0.9646814509235723
acc_score : 1.0

[drop out]
loss : 0.0003021926968358457
r2 score : 0.9986181057598148
acc_score : 1.0

[함수형 모델]
loss : 0.03300415724515915
r2 score : 0.8487817302870638
acc_score : 0.9444444444444444

[CPU]
loss : 0.029737671837210655
r2 score : 0.863625820247603
acc_score : 0.9444444444444444
걸린 시간 : 0.95 초
GPU 없다!~!

[GPU]
loss : 0.007030286826193333
r2 score : 0.9677699508450915
acc_score : 1.0
걸린 시간 : 1.99 초
GPU 돈다!~!

[DNN -> CNN ]
loss : 0.46283939480781555
acc : 0.89
r2 score : 0.5539211064417334
acc_score : 0.6666666666666666
걸린 시간 : 3.5 초

[LSTM]
loss : 0.4845461845397949
acc : 0.72
r2 score : 0.5421124849544849
acc_score : 0.6666666666666666
걸린 시간 : 4.22 초

[Conv1D]
loss : 0.4492349624633789
acc : 0.72
r2 score : 0.5661057912991788
acc_score : 0.6666666666666666
걸린 시간 : 3.28 초

"""



