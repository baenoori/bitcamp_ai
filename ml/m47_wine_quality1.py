import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

print('acc :', acc)      # acc : 0.6827272727272727
