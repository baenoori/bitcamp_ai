import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

#1. 데이터
x,y = load_iris(return_X_y=True)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, 
                                                    stratify=y,
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 
# 기준 점수
# acc :  [0.95833333 0.95833333 0.95833333 1.         1.        ]
# 평균 acc : 0.975


y_pre = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_pre)
print(y_test)
# [1 0 2 2 0 0 2 1 1 0 0 1 1 1 2 1 0 0 0 0 0 2 1 1 2 1 1 1 1 1]
# [1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 2 2 1 2 2 1 1 1 1]

acc = accuracy_score(y_test, y_pre)
print('cross_val_predict ACC :', acc)
# cross_val_predict ACC : 0.8666666666666667


# 기준 점수
# acc :  [0.95833333 0.95833333 0.95833333 1.         1.        ]
# 평균 acc : 0.975

