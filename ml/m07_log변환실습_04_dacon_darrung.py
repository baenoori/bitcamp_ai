from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#1. 데이터
path = "C:/ai5/_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # 점 하나(.) : 루트라는 뜻, index_col=0 : 0번째 열을 index로 취급해달라는 의미
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)      

train_csv.boxplot()
# df.plot.box()
# plt.show()
# hour_bef_visibility

print(train_csv.info())
print(train_csv.describe())

# df['target'].hist(bins=50)
# plt.show()

train_csv = train_csv.dropna()  # null 값 drop (삭제) 한다는 의미 

x = train_csv.drop(['count'], axis=1).copy()    # 행 또는 열 삭제 [count]라는 axis=1 열 (axis=0은 행)
print(x)    # [1328 rows x 9 columns]

y = train_csv['count']  # count 컬럼만 y에 넣음
print(y.shape)    # (1328,)

##### X population log 변환 #####
x['hour_bef_visibility'] = np.log1p(x['hour_bef_visibility']) # 지수 변환 : np.expm1
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
# log 변환 전 score : 0.7351536872725938
# y만 log 변환 score : 0.6916292630532399
# x만 log 변환 score : 0.7350811067099836
# x, y log 변환 score : 0.6916292630532399
"""


