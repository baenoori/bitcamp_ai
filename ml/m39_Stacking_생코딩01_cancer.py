# Pseudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾음
# 스태킹 : 모델을 돌려 나온거로 컬럼을 구성해서 새로운 데이터셋을 만듦
#       : 한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼을 만듦 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression     # 분류 ! 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1186,
                                                    stratify=y
                                                    )

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

train_list = []
test_list = []
models = [xgb, rf, cat]

for model in models:
    model.fit(x_train, y_train)
    y_pre = model.predict(x_train)
    y_test_pre = model.predict(x_test)
    
    train_list.append(y_pre)
    test_list.append(y_test_pre)

    score = accuracy_score(y_test, y_test_pre)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, score))


# XGBClassifier ACC : 0.9649
# RandomForestClassifier ACC : 0.9474
# CatBoostClassifier ACC : 0.9737
# numpy version 1.19.5

x_train_new = np.array(train_list).T
print(x_train_new.shape)        # (455, 3)


x_test_new = np.array(test_list).T
print(x_test_new.shape)        # (114, 3)

#2. 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred)
print('스태킹 결과 : ', score2)

# 스태킹 결과 : 0.9912280701754386


