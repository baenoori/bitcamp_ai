import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression     # 분류 ! 
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier, BaggingRegressor, VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

#1. 데이터 
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=4444,
                                                    # stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

train_list = []
test_list = []
models = [xgb, rf, cat]

for model in models:
    model.fit(x_train, y_train)
    y_pre = model.predict(x_train)
    y_test_pre = model.predict(x_test)
    
    train_list.append(y_pre)
    test_list.append(y_test_pre)

    score = r2_score(y_test, y_test_pre)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, score))

"""
XGBRegressor ACC : 0.8350
RandomForestRegressor ACC : 0.8128
CatBoostRegressor ACC : 0.8556
"""

x_train_new = np.array(train_list).T
print(x_train_new.shape)        # (455, 3)


x_test_new = np.array(test_list).T
print(x_test_new.shape)        # (114, 3)

#2. 모델
model2 = CatBoostRegressor(verbose=0)
model2.fit(x_train_new, y_train)
y_pred = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred)
print('스태킹 결과 : ', score2)

# 스태킹 결과 : 0.7905342254994817
