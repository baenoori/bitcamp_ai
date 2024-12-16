from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
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
from sklearn.datasets import load_digits


import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)

#1. 데이터
x, y = load_digits(return_X_y=True)
random_state = 777


from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=777, train_size=0.8,
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

# 이전 성능
#  0.9861111111111112

## PF
# 0.9805555555555555

