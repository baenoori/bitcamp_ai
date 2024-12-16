import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time 
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier, CatBoostRegressor

#1. 데이터
path = "C:/ai5/_data/kaggle/otto-group-product-classification-challenge/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_cav = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

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
        'iterations' : 100,
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

# {'target': np.float64(0.8177117000646412), 'params': {'colsample_bytree': np.float64(1.0), 'learning_rate': np.float64(0.1), 'max_bin': np.float64(9.0), 'max_depth': np.float64(10.0), 'min_child_samples': np.float64(107.72217612254735), 'min_child_weight': np.float64(1.0), 'num_leaves': np.float64(40.0), 'reg_alpha': np.float64(1.2050541054700847), 'reg_lambda': np.float64(-0.001), 'subsample': np.float64(1.0)}}
# 100 번 걸린 시간 : 194.08 초

# cat boost
# {'target': 0.798884938590821, 'params': {'bagging_temparature': 5.0, 'border_count': 137.64199921401354, 'depth': 12.0, 'l2_leaf_reg': 1.0, 'learning_rate': 0.1, 'random_strength': 2.2375316033563784}}
# 100 번 걸린 시간 : 1079.88 초


