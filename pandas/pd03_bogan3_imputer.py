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

imputer = SimpleImputer()
data2 = imputer.fit_transform(data)     # 평균 (디폴트)
print(data2)

imputer = SimpleImputer(strategy='mean')
data3 = imputer.fit_transform(data)     # 평균 
print(data3)

imputer = SimpleImputer(strategy='median')
data4 = imputer.fit_transform(data)     # 중위값
print(data4)

imputer = SimpleImputer(strategy='most_frequent')
data5 = imputer.fit_transform(data)     # 최빈값 (가장 자주 나오는 값)
print(data5)

imputer = SimpleImputer(strategy='constant', fill_value=777)
data6 = imputer.fit_transform(data)     # 상수, 특정값
print(data6)

#################

imputer = KNNImputer() # KNN 알고리즘으로 결측치 처리
data7 = imputer.fit_transform(data)
print(data7)

################

imputer = IterativeImputer() # 디폴트 : BayesianRidge 회귀모델
data8 = imputer.fit_transform(data)
print(data8)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0

# interpolate 에서는 맨 마지막 값과 첫번쨰값을  dfill과 ffill로 채워 줬는데 imputer에서는 알아서 처리해줌
