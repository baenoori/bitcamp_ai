import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso     # 분류 ! 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(777)

def create_multiclass_data_with_labels():
    X = np.random.rand(20,3)
    y = np.random.randint(0, 5, size=(20,3))
    
    X_df = pd.DataFrame(X, columns=['F1', 'f2', 'f3'])
    y_df = pd.DataFrame(y, columns=['l1', 'l2', 'l3'])
    
    return X_df.values, y_df.values

x, y = create_multiclass_data_with_labels()
print("x : " , x)
print("y : " , y)
print(x.shape)      # (20,3)
print(y.shape)      # (20,3)

#2. 모델
# model = RandomForestClassifier()
# model.fit(x,y)
# y_pre = model.predict(x)
# print(model.__class__.__name__, '스코어 : ', round(accuracy_score(y, y_pre), 4))   #RandomForestClassifier 스코어 :  0.0
# print(model.predict([[2,110,43]]))  # [[6 5 4]]

# model = LinearRegression()
# model.fit(x,y)
# y_pre = model.predict(x)
# print(model.__class__.__name__, '스코어 : ', round(accuracy_score(y, y_pre), 4))   # LinearRegression 스코어 :  1.9504
# print(model.predict([[2,110,43]]))  # [[217.03895934 116.61005634 -15.3943645 ]]

model = Ridge()
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   #Ridge 스코어 :  0.9461
print(model.predict([[2,110,43]]))  # [[103.06874446  -5.27432632 151.90625879]]

# model = XGBClassifier()         # error

# model.fit(x,y)
# y_pre = model.predict(x)
# print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   
# print(model.predict([[2,110,43]]))  

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

model = MultiOutputClassifier(LGBMClassifier())   
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre)), 4)   # MultiOutputClassifier 스코어 :  0 4
print(model.predict([[2,110,43]]))  # [[0 0 4]]

model = MultiOutputClassifier(CatBoostClassifier())      
model.fit(x,y)
y_pre = model.predict(x)
print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre.reshape(20,3)), 4))   #MultiOutputClassifier 스코어 :  0.0
print(model.predict([[2,110,43]]))  # [[[3 4 3]]]

#### catboost ####
# model = CatBoostClassifier(loss_function='MultiRMSE')
# model.fit(x,y)
# y_pre = model.predict(x)
# print(model.__class__.__name__, '스코어 : ', round(mean_absolute_error(y, y_pre), 4))   # CatBoostRegressor 스코어 :  0.0638
# print(model.predict([[2,110,43]]))  # [[138.21649371  32.99740595  67.8741709 ]]


