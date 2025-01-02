"""
그래프 그리기
1. value_counts => 쓰지 말기
2. np.unique의 return_counts => 쓰지말기

3. groupby 쓰기, count() 쓰기

# plt.bar 로 그리기 (quality 컬럼)

# 힌트
# 데이터 개수(y축) = 데이터갯수. ...

y: 데이터 개수
x : 3,4,5,6,7,8,9
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

path = "C:/ai5/_data/kaggle/wine/"

train = pd.read_csv(path + 'train.csv', index_col=0)

x = train.drop(['quality'], axis=1)
y = train['quality']

data = y.groupby(train['quality']).count()
print(data)
# quality
# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5

plt.bar(data.index, data.values)
plt.show()
