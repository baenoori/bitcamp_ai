# DNN -> CNN

from sklearn.datasets import load_digits
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



#1. 데이터 
x, y = load_digits(return_X_y=True)     # sklearn에서 데이터를 x,y 로 바로 반환

# print(x.shape, y.shape)     # (1797, 64) (1797,)

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

# x = x.reshape(1797,8,8)


n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)     # fit 제공됨
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 

# KFold
# acc :  [0.98333333 0.98888889 0.98328691 0.99442897 0.98607242] 
# 평균 acc : 0.9872

# StratifiedKFold
# acc :  [0.98611111 0.99444444 0.98885794 0.98885794 0.98050139] 
# 평균 acc : 0.9878







# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)
# print('loss :',loss[0])
# # print('acc :',round(loss[1],2))
# print('acc :',round(loss[1],2))

# y_pre = model.predict(x_test)
# r2 = r2_score(y_test, y_pre)
# print('r2 score :', r2)

# accuracy_score = accuracy_score(y_test,np.round(y_pre))
# print('acc_score :', accuracy_score)
# # print('걸린 시간 :', round(end-start, 2), '초')
# print("걸린 시간 :", round(end-start,2),'초')


"""
loss : 0.005590484477579594
r2 score : 0.9378835045441063
acc_score : 0.9555555555555556

[drop out]
loss : 0.005380750633776188
r2 score : 0.9402138763769592
acc_score : 0.9666666666666667

[함수형 모델]
loss : 0.00898907519876957
r2 score : 0.9001213876469093
acc_score : 0.9444444444444444

[CPU]
loss : 0.004038435406982899
r2 score : 0.9551284962296123
acc_score : 0.9722222222222222
걸린 시간 : 3.38 초
GPU 없다!~!

[GPU]
loss : 0.005989363417029381
r2 score : 0.9334515139632351
acc_score : 0.9666666666666667
걸린 시간 : 8.07 초
GPU 돈다!~!

[DNN -> CNN]
loss : 0.044570405036211014
acc : 0.99
r2 score : 0.9778775529022978
acc_score : 0.9888888888888889
걸린 시간 : 18.93 초

[LSTM]
loss : 0.31239330768585205
acc : 0.89
r2 score : 0.8404125980174719
acc_score : 0.8888888888888888
걸린 시간 : 39.1 초

[Conv1D]
loss : 0.09347185492515564
acc : 0.97
r2 score : 0.9438532828835641
acc_score : 0.9666666666666667
걸린 시간 : 17.45 초
"""

