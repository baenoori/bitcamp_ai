# 실습 
# 1. 아웃라이어 확인 
# 2. 아웃라이어 처리 
# 3. 47_1 이든 47_2든 수정해서 만들기 


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

path = "C:/ai5/_data/kaggle/wine/"

# 만들어보기 : y는 quality
#RF 디폴트로 

train = pd.read_csv(path + 'train.csv', index_col=0)
# print(train.head())
# print(train.info())
# print(train.isna().sum())

x = train.drop(['quality'], axis=1)
y = train['quality']

le = LabelEncoder()
x['type'] = le.fit_transform(x['type'])
print(le.transform(['red', 'white']))       # [0, 1]

############## outlier 확인 ##############
import matplotlib.pyplot as plt

def outliers(data):
    num_cols = len(data.columns)
    fig, axes = plt.subplots(num_cols, 1, figsize=(8, 6 * num_cols))  # subplot 생성
    if num_cols == 1:  # 열이 1개인 경우, axes를 리스트로 변환
        axes = [axes]

    for i, col in enumerate(data.columns):
        print(f"=== {col} ===")
        col_data = data[col]
        quartile_1, q2, quartile_3 = np.percentile(col_data, [25, 50, 75])
        
        print("1사분위:", quartile_1)
        print("중앙값(q2):", q2)
        print("3사분위:", quartile_3)
        
        iqr = quartile_3 - quartile_1
        print("IQR:", iqr)
        
        low_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        
        # 이상치 인덱스 출력
        outliers = col_data[(col_data < low_bound) | (col_data > upper_bound)]
        print("이상치:")
        print(outliers)
        
        # 박스플롯 시각화
        ax = axes[i]
        ax.boxplot(col_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax.axvline(low_bound, color='red', linestyle='--', label="Lower Bound")
        ax.axvline(upper_bound, color='blue', linestyle='--', label="Upper Bound")
        ax.set_title(f"Boxplot of {col}")
        ax.legend()

    # plt.tight_layout()
    # plt.show()

# 이상치 탐지 실행
outliers(x)

##########################################

################### 이상치 처리 ###################
from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.3)           # 30%를 이상치로 간주
           #  defualt = 0.1
for i in range(x.shape[1]):
    outliers = EllipticEnvelope(contamination=.2, support_fraction=0.8)
    a = x.iloc[:, i].values.reshape(-1,1)
    outliers.fit(a)
    results = outliers.predict(a)
    x.iloc[:, i] = np.where(results == -1, np.nan, x.iloc[:, i])    # 이상치를 NaN으로 처리

x.fillna(x.median(), inplace=True)
####################################################

le2 = LabelEncoder()
y = le2.fit_transform(y)
# y = y - 3
# print(pd.value_counts(y))
# 3    2416
# 2    1788
# 4     924
# 1     186
# 5     152
# 0      26
# 6       5

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2222, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pre = model.predict(x_test)
acc = accuracy_score(y_pre, y_test)

print('acc :', acc)      # acc : 0.7

# acc : 0.65
