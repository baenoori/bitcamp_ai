# DNN -> CNN

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional, Conv1D
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

#1. 데이터
path = "C:/ai5/_data/dacon/diabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]
print(x)    # [652 rows x 8 columns]
print(y.shape)    # (652, )

x = x.to_numpy()
x = x/255.

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)     # fit 제공됨
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 


# acc :  [0.70229008 0.71755725 0.73076923 0.77692308 0.78461538] 
# 평균 acc : 0.7424



### csv 파일 ###
# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# # print(y_submit)
# sampleSubmission_csv['Outcome'] = y_submit
# # print(sampleSubmission_csv)
# sampleSubmission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")


"""
loss : 0.17662686109542847
r2 score : 0.23671962825848636
acc_score : 0.7727272727272727

[drop out]
loss : 0.1760784089565277
r2 score : 0.23908975869665205
acc_score : 0.7424242424242424

[함수형 모델]
loss : 0.1697331964969635
r2 score : 0.2665102161316174
acc_score : 0.7575757575757576

[CPU]
loss : 0.18158185482025146
r2 score : 0.21530700200079433
acc_score : 0.7424242424242424
걸린 시간 : 1.43 초
GPU 없다!~!

[GPU]
loss : 0.17883329093456268
r2 score : 0.2271846824795315
acc_score : 0.7424242424242424
걸린 시간 : 4.23 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.5467386245727539
acc : 0.8
acc_score : 0.803030303030303
걸린 시간 : 6.57 초

[lSTM]
loss : 0.5381918549537659
acc : 0.74
acc_score : 0.7424242424242424
걸린 시간 : 8.43 초

[Conv1D]
loss : 0.5261954665184021
acc : 0.77
acc_score : 0.7727272727272727
걸린 시간 : 6.43 초
"""

