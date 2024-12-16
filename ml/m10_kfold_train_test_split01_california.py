from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, r2_score


#1. 데이터 
datasets  = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)   # [20640 rows x 8 columns]

df['target'] = datasets.target
print(df)   # [20640 rows x 9 columns]

# df.boxplot()
# df.plot.box()
# plt.show()
# population 열에 문제가 있음을 확인 <- 이상치 확인

print(df.info())
print(df.describe())

# df['Population'].boxplot()        # series 는 불가
# df['Population'].plot.box()
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

##### X population log 변환 #####
x['Population'] = np.log1p(x['Population']) # 지수 변환 : np.expm1
#################################

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, 
                                                    # stratify=y,
                                                    )
# ##### y population log 변환 #####
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
###################################

#2. 모델 구성
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,
                              min_samples_split=3,
                              )
# model = LinearRegression()
# model = SVC()

#3. 훈련

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 

y_pre = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = r2_score(y_test, y_pre)
print('cross_val_predict ACC :', acc)


"""
RandomForestRegressor 모델 
# log 변환 전 score : 0.6438535462775896
# y만 log 변환 score : 0.6593838015470808
# x만 log 변환 score : 0.6438535462775896
# x, y log 변환 score : 0.6593838015470808

LinearRegression 모델 
# log 변환 전 score : 0.6202298982993946
# y만 log 변환 score : 0.64220678377058
# x만 log 변환 score : 0.6203903724369276
# x, y log 변환 score : 0.6420214214797645


acc :  [0.65016278 0.65972062 0.64461787 0.67029479 0.66931079] 
평균 acc : 0.6588

--
acc :  [0.67690526 0.66347249 0.66120023 0.67455196 0.64423773] 
평균 acc : 0.6641
cross_val_predict ACC : 0.6646693827237501
"""

