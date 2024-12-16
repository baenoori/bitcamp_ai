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
path = "C:/ai5/_data/dacon/diabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]

x = x.to_numpy()
x = x/255.

random_state = 777

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=random_state, train_size=0.8,
                                                    stratify=y
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

# xgb 점수 : 0.7251908396946565
# hard 점수 : 0.7175572519083969
# soft 점수 : 0.732824427480916

