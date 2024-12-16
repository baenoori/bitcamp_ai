import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression     # 분류 ! 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
import pandas as pd

#1. 데이터 
path = "C:/ai5/_data/kaggle/playground-series-s4e1/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# 문자열 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']
x = x.to_numpy()
x = x/255.

random_state = 777

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=random_state, train_size=0.8,
                                                    stratify=y
                                                    )

print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

model = StackingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5,    
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pre = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('스태킹 acc :', accuracy_score(y_test, y_pre))

# model.score : 0.8639379525555185
# 스태킹 acc : 0.8639379525555185