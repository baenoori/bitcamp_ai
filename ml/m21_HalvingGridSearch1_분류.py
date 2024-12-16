import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV 
# 정식버전이 아니여서 line 5에 있은 enable_halving_search_cv import 필요
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y ,shuffle=True, random_state=3333, train_size=0.8,
    stratify=y
)
print(x_train.shape, y_train.shape) # (1437, 64) (1437,)
print(x_test.shape, y_test.shape)   # (360, 64) (360,)  


n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=3333)

parameters = [
     {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
      'max_depth': [3,4,5,6,8]},
     {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
      'subsample' : [0.6,0.7,0.8,0.9,1.0]},
     {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],'colsample_bytree' : [0.6,0.7,0.8,0.9,1.0]},
     {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma': [0,0.1,0.2,0.5,1.0]},
]   # 3*3*cv = 45번 

#2. 모델
model = HalvingGridSearchCV(XGBClassifier(
                                            # tree_method = 'gpu_hist'
                                            tree_method='hist',
                                            device = 'cuda:0' ,
                                            n_estimators=50,
                                            ), 
                            parameters, 
                            cv=kfold, 
                            verbose=1,  # 1: 이터레이터 내용만, 2: 훈련내용까지  
                            refit=True,     
                            #  n_jobs=-1,    
                            #  n_iter=10,
                            random_state=333,
                            factor=4,
                            min_resources=30,
                            max_resources=1437,
                            aggressive_elimination=True # 파라미터 제거를 factor-> factor + 1 
                            )

st = time.time()
model.fit(x_train, y_train, 
          verbose=False,
          eval_set=[(x_test,y_test)]
          )
et = time.time()

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)
print('best_score :', model.best_score_)    # train만 들어간 score, 훈련에서의 최고점
print('model.score', model.score(x_test, y_test))   # test로 뽑은 score
y_pre = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_pre))
y_pre_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', accuracy_score(y_test, y_pre_best))      # 요걸로 확인하기
print('걸린 시간 :', round(et-st, 2), '초')

# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device='cuda:0',
#               early_stopping_rounds=None, enable_categorical=False,
#               eval_metric=None, feature_types=None, gamma=None,
#               grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=4, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=50, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 4}
# best_score : 0.9433566433566434
# model.score 0.9638888888888889
# accuracy_score : 0.9638888888888889
# 최적 튠 ACC : 0.9638888888888889
# 걸린 시간 : 73.08 초

# factor = 3
# best_score : 0.9558581769108084
# model.score 0.9694444444444444
# accuracy_score : 0.9694444444444444
# 최적 튠 ACC : 0.9694444444444444
# 걸린 시간 : 148.61 초

exit()
import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)  
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)) # rank_test_score 기준 오름차순으로 정렬
print(pd.DataFrame(model.cv_results_).columns)
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_kernel', 'param_gamma', 'param_C', 'param_degree', 'params',
#        'split0_test_score', 'split1_test_score', 'split2_test_score',
#        'split3_test_score', 'split4_test_score', 'mean_test_score',
#        'std_test_score', 'rank_test_score'],
#       dtype='object'

path = "C:/ai5/_save/m15_GS_CV_01/"
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + "m17_RS_cv_result.csv")

