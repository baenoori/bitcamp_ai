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
from catboost import CatBoostClassifier, CatBoostRegressor

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

bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'depth': (2,12),
    'l2_leaf_reg' : (1,10),
    'bagging_temparature' : (0.0, 5.0),
    'border_count' : (32,255),
    'random_strength' : (1,10) 
}
def cat_hamsu(learning_rate, depth,
              l2_leaf_reg, bagging_temparature, 
              border_count, random_strength,
              ):            # 블랙박스 함수라고 함
    params = {
        'iterations' : 200,
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)),
        'l2_leaf_reg' : int(round(l2_leaf_reg)),
        # 'bagging_temparature' : max(min(bagging_temparature,1.0), 0.0),
        'border_count' : int(round(border_count)),
        'random_strength' :int(random_strength),
    }
    
    # cat_features = list(range(x_train.shape[1]))
    
    model = CatBoostRegressor(
                        **params,
                        task_type='GPU',                # GPU 사용 (기본값 : 'CPU')
                        devices='0',                    # 첫번째 CPU 사용 (기본값 : 모든 GPU 사용)
                        early_stopping_rounds=20,      # 조기 종료 (기본값 : None)
                        verbose=0,                     # 매 10번째 반복마다 출력 (기본값 : 100)
                        # eval_metric='F1',             # F1_score를 평가 지표로 설정, 다중분류(macro, micro)는 없음
                        # cat_features=cat_features       # 범주형 feature 지정 
                            )              
    
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],
            #   eval_metric='logloss',
              verbose = 0,
              )
    y_pre = model.predict(x_test)
    result = r2_score(y_test, y_pre)
    
    return result

bay = BayesianOptimization(
    f = cat_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # maximize 가 fit이라고 생각
et = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간 :', round(et-st, 2), '초')

# {'target': np.float64(0.7282082585811853), 'params': {'colsample_bytree': np.float64(0.7716455435126173), 'learning_rate': np.float64(0.07316612206200168), 'max_bin': np.float64(17.28879000553614), 'max_depth': np.float64(5.312371620396405), 'min_child_samples': np.float64(80.05714672806884), 'min_child_weight': np.float64(3.3668796217250136), 'num_leaves': np.float64(25.672483010759535), 'reg_alpha': np.float64(4.880901405662105), 'reg_lambda': np.float64(2.453278548450057), 'subsample': np.float64(0.9367446772294055)}}
# 500 번 걸린 시간 : 166.44 초

# cat boost
# {'target': 0.7968146155046634, 'params': {'bagging_temparature': 5.0, 'border_count': 130.94031977454074, 'depth': 12.0, 'l2_leaf_reg': 1.0, 'learning_rate': 0.1, 'random_strength': 6.914123153919017}}
# 100 번 걸린 시간 : 182.65 초
