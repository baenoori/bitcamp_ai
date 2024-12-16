import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso     # 분류 ! 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

#1. 데이터
x, y = load_linnerud(return_X_y=True)
print(x.shape, y.shape)     # (20, 3) (20, 3)
print(x)
print(y)

#2. 모델
model = RandomForestRegressor()
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # RandomForestRegressor 스코어 :  3.5262
print(model.predict([[2,110,43]]))  # [[159.42  34.86  62.92]]

model = LinearRegression()
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # LinearRegression 스코어 :  7.4567
print(model.predict([[2,110,43]]))  # [[187.33745435  37.08997099  55.40216714]]

model = Ridge()
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   #Ridge 스코어 :  7.4569
print(model.predict([[2,110,43]]))  # [[187.32842123  37.0873515   55.40215097]]

model = XGBRegressor()
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # XGBRegressor 스코어 :  0.0008
print(model.predict([[2,110,43]]))  # [[138.0005    33.002136  67.99897 ]]

# model = CatBoostRegressor()        # error
# model.fit(x,y)
# y_pre = model.predict(x)
# print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # XGBRegressor 스코어 :  0.0008
# print(model.predict([[2,110,43]]))  # [[138.0005    33.002136  67.99897 ]]

# model = LGBMRegressor()       # error
# model.fit(x,y)
# y_pre = model.predict(x)
# print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # XGBRegressor 스코어 :  0.0008
# print(model.predict([[2,110,43]]))  # [[138.0005    33.002136  67.99897 ]]

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

model = MultiOutputRegressor(LGBMRegressor())   
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # MultiOutputRegressor 스코어 :  8.91
print(model.predict([[2,110,43]]))  # [[178.6  35.4  56.1]]

model = MultiOutputRegressor(CatBoostRegressor())      
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # MultiOutputRegressor 스코어 :  0.2154
print(model.predict([[2,110,43]]))  # [[138.97756017  33.09066774  67.61547996]]


#### catboost ####
model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # CatBoostRegressor 스코어 :  0.0638
print(model.predict([[2,110,43]]))  # [[138.21649371  32.99740595  67.8741709 ]]



