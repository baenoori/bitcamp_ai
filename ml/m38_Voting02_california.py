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


# 2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

# model = XGBRegressor()
model = VotingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
    # voting='hard',  # 디폴트
    # voting='soft',
    
)

# 3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_pre = model.predict(x_test)
acc = r2_score(y_test, y_pre)
print('acc y_pre : ', acc)

# xgb 점수 : 0.0.834965482631102
# hard 점수 : 0.8485294055292903
# soft 점수 :

