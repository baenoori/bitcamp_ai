import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression     # 분류 ! 
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier, BaggingRegressor, VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor

#1. 데이터 
x, y = load_diabetes(return_X_y=True)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1186,
                                                    )

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

model = StackingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv=5,    
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pre = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('스태킹 acc :', r2_score(y_test, y_pre))

# model.score : 0.3435846402880731
# 스태킹 acc : 0.3435846402880731


## PE
# 0.2797805152624794
