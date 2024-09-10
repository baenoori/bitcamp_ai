import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time 
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
random_state = 777

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=random_state, train_size=0.8,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
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

early_stop = xgb.callback.EarlyStopping(
    rounds = 20, 
    # metric_name = 'logloss',   # eval_metric 과 동일하게 
    data_name = 'validation_0',
    # save_best = True,     # 아래 에러 생김
    # AttributeError: `best_iteration` is only defined when early stopping is used.
)

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
    model = XGBClassifier(**params, n_jobs=-1, callbacks=[early_stop])
    
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],
            #   eval_metric='logloss',
              verbose = 0,
              )
    y_pre = model.predict(x_test)
    result = accuracy_score(y_test, y_pre)
    
    return result

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # maximize 가 fit이라고 생각
et = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간 :', round(et-st, 2), '초')

# {'target': np.float64(0.9824561403508771), 'params': {'learning_rate': np.float64(0.0866510891852539), 'max_depth': np.float64(3.4468381398272507)}}
# 100 번 걸린 시간 : 10.37 초

# {'target': np.float64(0.9912280701754386), 'params': {'colsample_bytree': np.float64(0.7483578028902041), 'learning_rate': np.float64(0.07457430294179365), 'max_bin': np.float64(26.07999118063211), 'max_depth': np.float64(8.890832601780346), 'min_child_samples': np.float64(91.72651761417472), 'min_child_weight': np.float64(6.298933757300313), 'num_leaves': np.float64(25.658170963372612), 'reg_alpha': np.float64(0.24284942460706926), 'reg_lamba': np.float64(1.3138906868392526), 'subsample': np.float64(0.5891212452707173)}}
# 100 번 걸린 시간 : 11.4 초
