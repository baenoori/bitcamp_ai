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
                                                    # stratify=y
                                                    )


# 2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

# model = XGBClassifier()
model = VotingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
    # voting='hard',  # 디폴트
    voting='soft',
)

# 3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)
print('acc y_pre : ', acc)

# xgb 점수 : 0.8642712151967764
# hard 점수 : 0.8646650710455358
# soft 점수 : 0.8645438846305329

