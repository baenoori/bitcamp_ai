from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

#1. 데이터
# x,y = load_iris(return_X_y=True)
# print(x.shape, y.shape)      # (150, 4) (150,)
datasets = load_iris()
x = datasets.data
y = datasets.target

print(datasets)

df = pd.DataFrame(x, columns=datasets.feature_names)
# print(df)
df['Target'] = y
print(df)

print("========================= 상관계수 히트맵 ============================")
print(df.corr())


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

