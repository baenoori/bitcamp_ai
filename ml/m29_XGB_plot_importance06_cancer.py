import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3377,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
early_stop = xgb.callback.EarlyStopping(
    rounds = 50, 
    metric_name = 'logloss',   # eval_metric 과 동일하게 
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
    eval_metric = 'logloss',      # 다중분류:mlogloss,merror 이진분류:logloss,error / 2.1.1 버전에서 fit에서 모델파라미터로 이동
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

# 최종점수 : 0.9912280701754386
# acc_score : 0.9912280701754386

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
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

random_state = 1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, 
                                                    stratify=y
                                                    )

#2. 모델구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

print("random_state :", random_state)
for model in models:
    model.fit(x_train, y_train)
    print("=====================", model.__class__.__name__, "=====================")
    print('acc :', model.score(x_test,y_test))
    print(model.feature_importances_)

import matplotlib.pylab as plt
import numpy as np

# print(model)    # 파라미터 나옴

# def plot_feature_importance_datasets(model):    # model : XGboost
#     n_features = datasets.data.shape[1]     # 4
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
#     plt.yticks(np.arange(n_features), model.feature_importances_)
#     plt.xlabel("feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)

# for i, model in enumerate(models):
#     model.fit(x_train, y_train)
#     print("=====================", model.__class__.__name__, "=====================")
#     print('acc :', model.score(x_test,y_test))
#     print(model.feature_importances_)
#     plt.subplot(2,2,i+1)
#     plot_feature_importance_datasets(model)

# plt.rc('xtick', labelsize=5)
# plt.rc('ytick', labelsize=5)
# plt.tight_layout()  # 간격 안겹치게 
# plt.show()

# plt.subplot(2,2,1)
# plot_feature_importance_datasets(model1)

# plt.subplot(2,2,2)
# plot_feature_importance_datasets(model2)

# plt.subplot(2,2,3)
# plot_feature_importance_datasets(model3)

# plt.subplot(2,2,4)
# plot_feature_importance_datasets(model4)

from xgboost.plotting import plot_importance 

plot_importance(model)
plt.show()
