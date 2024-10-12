import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ])

data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

imputer = IterativeImputer()    # 디폴트 : BayesianRidge 회귀모델 
data1 = imputer.fit_transform(data)
print(data1)

imputer = IterativeImputer(estimator=DecisionTreeRegressor())    # 디폴트 : BayesianRidge 회귀모델 
data2 = imputer.fit_transform(data)
print(data2)
# decisionTree 알고리즘으로 찾음
# [[ 2.  2.  2.  4.]
#  [ 6.  4.  4.  4.]
#  [ 6.  4.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]]

imputer = IterativeImputer(estimator=RandomForestRegressor())    # 디폴트 : BayesianRidge 회귀모델 
data3 = imputer.fit_transform(data)
print(data3)
# [[ 2.    2.    2.    4.8 ]
#  [ 4.22  4.    4.    4.  ]
#  [ 6.    3.8   6.    4.8 ]
#  [ 8.    8.    8.    8.  ]
#  [10.    6.58 10.    6.72]]

imputer = IterativeImputer(estimator=xgb.XGBRegressor())    # 디폴트 : BayesianRidge 회귀모델 
data4 = imputer.fit_transform(data)
print(data4)
# [[ 2.          2.          2.          4.00096321]
#  [ 2.00112057  4.          4.          4.        ]
#  [ 6.          4.00000906  6.          4.00096321]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.99906492 10.          7.99903679]]

