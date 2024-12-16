import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

#1. 데이터
datasets = load_breast_cancer()
x=datasets.data
y=datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
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
