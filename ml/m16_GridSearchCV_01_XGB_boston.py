import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import time
import xgboost as xgb


#1. 데이터
x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y ,shuffle=True, random_state=3333, train_size=0.8,
    # stratify=y
)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=3333)

parameters = [
    {'n_jobs':[-1,], 'n_estimators': [100,500], 'max_depth':[6,10,12], 'min_sample_leaf':[3,10]}, # 12
    {'n_jobs':[-1,], 'max_depth':[6,8, 10,12], 'min_sample_leaf':[3,5,7,10]}, # 16
    {'n_jobs':[-1,], 'min_sample_leaf':[3,5,7,10], 'min_sample_split':[2,3,5,10]}, # 16
    {'n_jobs':[-1,], 'min_sample_split':[2,3,5,10]}, # 4
]

#2. 모델
model = GridSearchCV(xgb.XGBRegressor(), parameters, cv=kfold, 
                     verbose=1, 
                     refit=True,    # 가장 좋은 모델 한번 더 돌리기 
                     n_jobs=-1,     # cpu 모든 코어 다 쓰기 (24개의 코어) 
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


# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, min_sample_leaf=3, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=100,
#              n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'max_depth': 6, 'min_sample_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}
# best_score : 0.8648259424237604
# model.score 0.8760396431554781
# accuracy_score : 0.8760396431554781
# 최적 튠 ACC : 0.8760396431554781
# 걸린 시간 : 5.33 초
