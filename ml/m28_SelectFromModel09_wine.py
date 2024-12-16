import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3377,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
early_stop = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'logloss',   # eval_metric 과 동일하게 
    data_name = 'validation_0',
    # save_best = True,     # 아래 에러 생김
    # AttributeError: `best_iteration` is only defined when early stopping is used.
)

model = XGBClassifier(
    n_estimators = 500, 
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    rag_alpha = 0,  # L1 규제 - 가중치 규제
    reg_lambda = 1, # L2 규제 - 가중치 규제
    # eval_metric = 'logloss',      # 다중분류:mlogloss,merror 이진분류:logloss,error / 2.1.1 버전에서 fit에서 모델파라미터로 이동
    callbacks = [early_stop],
    random_state = 3377,
) 

#3. 훈련
model.fit(x_train, y_train, 
          eval_set = [(x_test, y_test)],
          verbose=1,
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_pre = model.predict(x_test)
acc= accuracy_score(y_test, y_pre)
print('acc_score :', acc)

# 최종점수 : 0.9722222222222222
# acc_score : 0.9722222222222222

print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)     # 오름차순 정렬
print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBClassifier(
                                n_estimators = 500, 
                                max_depth = 6,
                                gamma = 0,
                                min_child_weight = 0,
                                subsample = 0.4,
                                rag_alpha = 0,  # L1 규제 - 가중치 규제
                                reg_lambda = 1, # L2 규제 - 가중치 규제
                                # eval_metric = 'logloss',      # 다중분류:mlogloss,merror 이진분류:logloss,error / 2.1.1 버전에서 fit에서 모델파라미터로 이동
                                # callbacks = [early_stop],
                                random_state = 3377,)

    select_model.fit(select_x_train, y_train,
                     eval_set = [(select_x_test, y_test)],
                     verbose=False,
                     )
    select_y_pre = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pre)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))

# Trech=0.002, n=13, ACC: 97.22%
# Trech=0.003, n=12, ACC: 97.22%
# Trech=0.012, n=11, ACC: 97.22%
# Trech=0.019, n=10, ACC: 97.22%
# Trech=0.037, n=9, ACC: 94.44%
# Trech=0.039, n=8, ACC: 94.44%
# Trech=0.054, n=7, ACC: 94.44%
# Trech=0.091, n=6, ACC: 97.22%
# Trech=0.127, n=5, ACC: 97.22%
# Trech=0.142, n=4, ACC: 97.22%
# Trech=0.143, n=3, ACC: 97.22%
# Trech=0.158, n=2, ACC: 97.22%
# Trech=0.175, n=1, ACC: 66.67%

