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

#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=4444,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


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

# xgb 점수 : 0.9649122807017544
# hard 점수 : 0.956140350877193
# soft 점수 : 0.956140350877193

