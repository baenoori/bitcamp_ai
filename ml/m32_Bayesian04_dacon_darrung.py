import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time 
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "C:/ai5/_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # 점 하나(.) : 루트라는 뜻, index_col=0 : 0번째 열을 index로 취급해달라는 의미
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)      

train_csv = train_csv.dropna()  # null 값 drop (삭제) 한다는 의미 

x = train_csv.drop(['count'], axis=1).copy()    # 행 또는 열 삭제 [count]라는 axis=1 열 (axis=0은 행)

y = train_csv['count']  # count 컬럼만 y에 넣음

random_state = 777

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=random_state, train_size=0.8,
                                                    # stratify=y
                                                    )

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델

early_stop = xgb.callback.EarlyStopping(
    rounds = 20, 
    # metric_name = 'logloss',   # eval_metric 과 동일하게 
    data_name = 'validation_0',
    # save_best = True,     # 아래 에러 생김
    # AttributeError: `best_iteration` is only defined when early stopping is used.
)

bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3,10),
    'num_leaves' : (24,40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),    
}

def xgb_hamsu(learning_rate, max_depth,
              num_leaves, min_child_samples, 
              min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha,
              ):            # 블랙박스 함수라고 함
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)),10),
        'reg_lambda' : max(reg_lambda,0),
        'reg_alpha' : reg_alpha,  
    }
    model = XGBRegressor(**params, n_jobs=-1, callbacks=[early_stop])
    
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],
            #   eval_metric='logloss',
              verbose = 0,
              )
    y_pre = model.predict(x_test)
    result = r2_score(y_test, y_pre)
    
    return result

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 500
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # maximize 가 fit이라고 생각
et = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간 :', round(et-st, 2), '초')

# {'target': np.float64(0.7282082585811853), 'params': {'colsample_bytree': np.float64(0.7716455435126173), 'learning_rate': np.float64(0.07316612206200168), 'max_bin': np.float64(17.28879000553614), 'max_depth': np.float64(5.312371620396405), 'min_child_samples': np.float64(80.05714672806884), 'min_child_weight': np.float64(3.3668796217250136), 'num_leaves': np.float64(25.672483010759535), 'reg_alpha': np.float64(4.880901405662105), 'reg_lambda': np.float64(2.453278548450057), 'subsample': np.float64(0.9367446772294055)}}
# 500 번 걸린 시간 : 166.44 초



