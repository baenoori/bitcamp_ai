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
from sklearn.metrics import accuracy_score, r2_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression     # 분류 ! 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

#1. 데이터
path = "C:/ai5/_data/dacon/diabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]

x = x.to_numpy()
x = x/255.

random_state = 777

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=random_state, train_size=0.8,
                                                    stratify=y
                                                    )
#2. 모델
# model = DecisionTreeClassifier()
# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators = 100,
#                           n_jobs = -1,
#                           random_state = 4444,
#                         #   bootstrap=True,       # 디폴트, 중복 허용
#                           bootstrap=False,       # 중복 허용 안함
#                           )
# model = LogisticRegression()
# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators = 100,
#                           n_jobs = -1,
#                           random_state = 4444,
#                         #   bootstrap=True,       # 디폴트, 중복 허용
#                           bootstrap=False,       # 중복 허용 안함
#                           )
model = BaggingClassifier(RandomForestClassifier(),
                          n_estimators = 100,
                          n_jobs = -1,
                          random_state = 4444,
                          bootstrap=True,       # 디폴트, 중복 허용
                          # bootstrap=False,       # 중복 허용 안함
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('최종 점수 :', result)

y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)
print('acc :', acc)

# RandomForestClassifier - bagging - bootstrap=False
# 최종 점수 : 0.7099236641221374
# acc : 0.7099236641221374

# RandomForestClassifier - bagging - bootstrap=True
# 최종 점수 : 0.7022900763358778
# acc : 0.7022900763358778