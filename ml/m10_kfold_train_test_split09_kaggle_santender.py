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
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, r2_score


#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)


print(train_csv.info())
print(train_csv.describe())


train_csv.boxplot()
# train_csv.plot.box()
# plt.show()
# var_45, var_74, var_117, var_120

# df['target'].hist(bins=50)
# plt.show()


x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, 
                                                    stratify=y,
                                                    )
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
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 

y_pre = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_pre)
print('cross_val_predict ACC :', acc)


"""
RandomForestRegressor 모델 
# log 변환 전 score : 0.04897865480329777
# y만 log 변환 score : 0.04895696135901162

KFold
# acc :  [0.913725 0.910275 0.91365  0.909875 0.913625] 
# 평균 acc : 0.9122

StratifiedKFold
acc :  [0.9113   0.9122   0.912525 0.912475 0.911775] 
평균 acc : 0.9121

acc :  [0.91115625 0.91284375 0.9125625  0.911125   0.91140625] 
평균 acc : 0.9118
cross_val_predict ACC : 0.909725
"""

