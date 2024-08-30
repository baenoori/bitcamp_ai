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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
import time
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

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

x_train, x_test, y_train, y_test = train_test_split(
    x, y ,shuffle=True, random_state=3333, train_size=0.8,
    # stratify=y
)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=3333)

parameters = [
    {'n_jobs':[-1,], 'n_estimators': [100,500], 'max_depth':[6,10,12], 'min_sample_leaf':[3,10], 'learning_rate':[0.1,0.01,0.001,0.005]},
    {'n_jobs':[-1,], 'max_depth':[6,8, 10,12], 'min_sample_leaf':[3,5,7,10]}, # 16
    {'n_jobs':[-1,], 'min_sample_leaf':[3,5,7,10], 'min_sample_split':[2,3,5,10], 'learning_rate':[0.01,0.001,0.005]},
    {'n_jobs':[-1,], 'min_sample_split':[2,3,5,10]}, 
]

#2. 모델
model = RandomizedSearchCV(xgb.XGBRegressor(), parameters, cv=kfold, 
                     verbose=1, 
                     refit=True,    # 가장 좋은 모델 한번 더 돌리기 
                     n_jobs=-1,     # cpu 모든 코어 다 쓰기 (24개의 코어) 
                     n_iter=10,
                     random_state=3333
                     )

st = time.time()
model.fit(x_train, y_train)
et = time.time()

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)
print('best_score :', model.best_score_)    # train만 들어간 score, 훈련에서의 최고점
print('model.score', model.score(x_test, y_test))   # test로 뽑은 score
y_pre = model.predict(x_test)
print('accuracy_score :', r2_score(y_test,y_pre))
y_pre_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', r2_score(y_test, y_pre_best))      # 요걸로 확인하기
print('걸린 시간 :', round(et-st, 2), '초')

# Grid
# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, min_sample_leaf=3, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=500,
#              n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'max_depth': 6, 'min_sample_leaf': 3, 'n_estimators': 500, 'n_jobs': -1}
# best_score : 0.9993075132369995
# model.score 0.9993106722831726
# accuracy_score : 0.9993106722831726
# 최적 튠 ACC : 0.9993106722831726
# 걸린 시간 : 10.65 초

# Random
# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, min_sample_leaf=3, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=500,
#              n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'n_jobs': -1, 'n_estimators': 500, 'min_sample_leaf': 3, 'max_depth': 6, 'learning_rate': 0.1} 
# best_score : 0.9994479537010192
# model.score 0.9994959235191345
# accuracy_score : 0.9994959235191345
# 최적 튠 ACC : 0.9994959235191345
# 걸린 시간 : 4.07 초
