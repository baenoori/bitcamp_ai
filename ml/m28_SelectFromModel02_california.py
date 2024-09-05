import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3377,
                                                    # stratify=y
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

model = XGBRegressor(
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
acc= r2_score(y_test, y_pre)
print('acc_score :', acc)

# 최종점수 : 0.7991229273622688
# acc_score : 0.7991229273622688

print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)     # 오름차순 정렬
print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBRegressor(
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
    score = r2_score(y_test, select_y_pre)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))



# Trech=0.052, n=8, ACC: 78.68%
# Trech=0.054, n=7, ACC: 78.54%
# Trech=0.068, n=6, ACC: 79.07%
# Trech=0.078, n=5, ACC: 78.65%
# Trech=0.121, n=4, ACC: 78.75%
# Trech=0.128, n=3, ACC: 60.99%
# Trech=0.143, n=2, ACC: 45.43%
# Trech=0.356, n=1, ACC: 47.07%


