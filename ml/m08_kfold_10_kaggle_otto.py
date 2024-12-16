from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

#1. 데이터
path = "C:/ai5/_data/kaggle/otto-group-product-classification-challenge/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_cav = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])

print(train_csv.info())
print(train_csv.describe())


train_csv.boxplot()
# train_csv.plot.box()
# plt.show()
# feat_24, feat_73

# df['target'].hist(bins=50)
# plt.show()


x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# y = pd.get_dummies(y)

##### X population log 변환 ##### 
x['feat_24'] = np.log1p(x['feat_24']) # 지수 변환 : np.expm1
x['feat_73'] = np.log1p(x['feat_73']) # 지수 변환 : np.expm1
#################################

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)     # fit 제공됨
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 




"""
RandomForestRegressor 모델 
# log 변환 전 score : 0.2628684887538744
# y만 log 변환 score : 0.2626253351793317
# x만 log 변환 score : 0.2628684887538744
# x, y log 변환 score : 0.2626253351793317

KFold
# acc :  [0.77989657 0.78935036 0.77666451 0.7849697  0.78238384] 
# 평균 acc : 0.7827

StratifiedKFold
acc :  [0.78312864 0.78821913 0.78312864 0.78044444 0.77931313] 
평균 acc : 0.7828
"""


