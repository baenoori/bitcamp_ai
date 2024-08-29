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
path = 'C:/ai5/_data/kaggle/bike-sharing-demand/'   # 절대경로 , 파이썬에서 \\a는 '\a'로 취급 특수문자 쓸 때 주의

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 'datatime'열은 인덱스 취급, 데이터로 X
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)  

train_csv = train_csv.dropna()

print(train_csv.info())
print(train_csv.describe())


train_csv.boxplot()
# train_csv.plot.box()
# plt.show()
#  casual,  registered

# df['target'].hist(bins=50)
# plt.show()

x = train_csv.drop(['count'], axis = 1) # [0, 0] < list (2개 이상은 리스트)
y = train_csv['count']

##### X population log 변환 ##### 
x['casual'] = np.log1p(x['casual']) # 지수 변환 : np.expm1
x['registered'] = np.log1p(x['registered']) # 지수 변환 : np.expm1
#################################

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)

# print(x_train.shape, y_train.shape) # (1313, 9) (1313,)

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
# log 변환 전 score : 0.9918646671250175
# y만 log 변환 score : 0.9947659095426319
# x만 log 변환 score : 0.9918983320785206
# x, y log 변환 score : 0.994768732848243

acc :  [0.99243636 0.99383562 0.99251795 0.99201344 0.99359428] 
평균 acc : 0.9929
"""


