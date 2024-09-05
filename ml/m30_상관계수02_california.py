from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pandas as pd

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
df['Target'] = y
print(df)

print("========================= 상관계수 히트맵 ============================")
print(df.corr())
#               MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude    Target
# MedInc      1.000000 -0.119034  0.326895  -0.062040    0.004834  0.018766 -0.079809  -0.015176  0.688075
# HouseAge   -0.119034  1.000000 -0.153277  -0.077747   -0.296244  0.013191  0.011173  -0.108197  0.105623
# AveRooms    0.326895 -0.153277  1.000000   0.847621   -0.072213 -0.004852  0.106389  -0.027540  0.151948
# AveBedrms  -0.062040 -0.077747  0.847621   1.000000   -0.066197 -0.006181  0.069721   0.013344 -0.046701
# Population  0.004834 -0.296244 -0.072213  -0.066197    1.000000  0.069863 -0.108785   0.099773 -0.024650
# AveOccup    0.018766  0.013191 -0.004852  -0.006181    0.069863  1.000000  0.002366   0.002476 -0.023737
# Latitude   -0.079809  0.011173  0.106389   0.069721   -0.108785  0.002366  1.000000  -0.924664 -0.144160
# Longitude  -0.015176 -0.108197 -0.027540   0.013344    0.099773  0.002476 -0.924664   1.000000 -0.045967
# Target      0.688075  0.105623  0.151948  -0.046701   -0.024650 -0.023737 -0.144160  -0.045967  1.000000

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True, # 표 안에 수치 명시
            cbar=True   # 사이드 바
            )
plt.show()



