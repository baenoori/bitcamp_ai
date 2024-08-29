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
from sklearn.svm import SVC

#1. 데이터 
datasets  = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)   # [20640 rows x 8 columns]

df['target'] = datasets.target
print(df)   # [20640 rows x 9 columns]

df.boxplot()
# df.plot.box()
# plt.show()
# x 이상치 x , y 만 log

print(df.info())
print(df.describe())

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

##### X population log 변환 #####
# x['TAX'] = np.log1p(x['TAX']) # 지수 변환 : np.expm1
# x['B'] = np.log1p(x['B']) # 지수 변환 : np.expm1
#################################

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)

##### y population log 변환 #####
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
###################################

#2. 모델 구성
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,
                              min_samples_split=3,
                              )

#3. 훈련

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

scores = cross_val_score(model, x, y, cv=kfold)     # fit 제공됨
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 


# #4. 평가, 예측
# score = model.score(x_test, y_test)     # r2_score
# print('score :', score)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2_score : ', r2)


"""
RandomForestRegressor 모델 
# log 변환 전 score : 0.8666074992549043
# y만 log 변환 score : 0.8357494567213853
# x만 log 변환 score : 0.8663854004220358
# x, y log 변환 score : 0.8352951067023585

acc :  [0.40641002 0.37224969 0.50227657 0.37385499 0.44710984] 
평균 acc : 0.4204

"""


