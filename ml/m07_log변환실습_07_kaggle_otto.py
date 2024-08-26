from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

y = pd.get_dummies(y)

##### X population log 변환 ##### 
x['feat_24'] = np.log1p(x['feat_24']) # 지수 변환 : np.expm1
x['feat_73'] = np.log1p(x['feat_73']) # 지수 변환 : np.expm1
#################################

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)

print(x_train.shape, y_train.shape) # (1313, 9) (1313,)

##### y population log 변환 #####
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
###################################

#2. 모델 구성
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,
                              min_samples_split=3,
                              )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)     # r2_score
print('score :', score)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)


"""
RandomForestRegressor 모델 
# log 변환 전 score : 0.2628684887538744
# y만 log 변환 score : 0.2626253351793317
# x만 log 변환 score : 0.2628684887538744
# x, y log 변환 score : 0.2626253351793317
"""


