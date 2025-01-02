import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

path = "C:/ai5/_data/kaggle/wine/"

# 만들어보기 : y는 quality

train = pd.read_csv(path + 'train.csv', index_col=0)
# print(train.head())
# print(train.info())
# print(train.isna().sum())

x = train.drop(['quality'], axis=1)
y = train['quality']

le = LabelEncoder()
x['type'] = le.fit_transform(x['type'])
print(le.transform(['red', 'white']))       # [0, 1]

# le2 = LabelEncoder()
# y = le2.fit_transform(y)
# y = y - 3
print(y.value_counts().sort_index())
# quality
# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5

########################################################
# [실습] y의 클래스를 7개에서 5~3개로 줄여서 성능 비교
y = y.copy()

# for index, value in enumerate(y):
# train = train[(train['quality']!=9) & (train['quality']!=3)]
# train = train[(train['quality']!=9) & (train['quality']!=3) & (train['quality']!=4)]
# train = train[(train['quality']!=9) & (train['quality']!=3) & (train['quality']!=4) & (train['quality']!=8)]

train['quality'] = train['quality'].replace(3, 5)
train['quality'] = train['quality'].replace(4, 5)
train['quality'] = train['quality'].replace(9, 7)
train['quality'] = train['quality'].replace(8, 7)

########################################################

# for문
# for i, v in enumerate(y):
#     if v <= 4:
#         y[i] = 0
#     elif v == 5:
#         y[i] = 1
#     else:
#         y[i] = 2

########################################################

x = train.drop(['quality'], axis=1)
y = train['quality']

print(y.value_counts().sort_index())

le = LabelEncoder()
x['type'] = le.fit_transform(x['type'])

y = y - 5

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9978, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=4525,
)

model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose=0)

y_pre = model.predict(x_test)
acc = accuracy_score(y_pre, y_test)

print('acc :', acc)     
print('F1 :',f1_score(y_pre, y_test, average='macro'))     

data = y.groupby(train['quality']).count()
# print(data)

plt.bar(data.index, data.values)
plt.show()

"""
기본
acc : 0.6827272727272727

5개로 줄이기
acc : 0.6846435100548446

4개로 줄이기
acc : 0.678030303030303

3개로 줄이기
acc : 0.7076023391812866

3개로 합치기
acc : 0.7127272727272728
F1 : 0.7040043067018052
"""

