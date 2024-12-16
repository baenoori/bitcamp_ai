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

from catboost import CatBoostClassifier, CatBoostRegressor


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
    'depth': (2,12),
    'l2_leaf_reg' : (1,10),
    'bagging_temparature' : (0, 5),
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
        'bagging_temparature' : bagging_temparature,
        'border_count' : int(round(border_count)),
        'random_strength' :int(random_strength),
    }
    
    # cat_features = list(range(x_train.shape[1]))
    
    model = CatBoostClassifier(
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
    result = accuracy_score(y_test, y_pre)
    
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

# {'target': np.float64(0.9824561403508771), 'params': {'learning_rate': np.float64(0.0866510891852539), 'max_depth': np.float64(3.4468381398272507)}}
# 100 번 걸린 시간 : 10.37 초

# {'target': np.float64(0.9912280701754386), 'params': {'colsample_bytree': np.float64(0.7483578028902041), 'learning_rate': np.float64(0.07457430294179365), 'max_bin': np.float64(26.07999118063211), 'max_depth': np.float64(8.890832601780346), 'min_child_samples': np.float64(91.72651761417472), 'min_child_weight': np.float64(6.298933757300313), 'num_leaves': np.float64(25.658170963372612), 'reg_alpha': np.float64(0.24284942460706926), 'reg_lamba': np.float64(1.3138906868392526), 'subsample': np.float64(0.5891212452707173)}}
# 100 번 걸린 시간 : 11.4 초

# cat boost
# {'target': 1.0, 'params': {'bagging_temparature': 0.0, 'border_count': 179.42463648594958, 'depth': 2.0, 'l2_leaf_reg': 10.0, 'learning_rate': 0.1, 'random_strength': 5.052896940862996}}      
# 100 번 걸린 시간 : 128.43 초