import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

#1. 데이터 
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y ,shuffle=True, random_state=3333, train_size=0.8,
    stratify=y
)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=3333)

parameters = [
    {"C":[1,10,100,1000], "kernel":['linear', 'sigmoid'], 'degree':[3,4,5]}, # 24번 돌아감
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[0.001,0.0001]},      # 6번 돌아감 
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.01,0.001,0.0001], 'degree':[3,4]}    # 24번 돌아감
]   # 총 54번 

#2. 모델
# model = GridSearchCV(SVC(), parameters, cv=kfold, 
#                      verbose=1, 
#                      refit=True,    # 가장 좋은 모델 한번 더 돌리기 
#                      n_jobs=-1,     # cpu 모든 코어 다 쓰기 (24개의 코어) 
#                      )   

model = RandomizedSearchCV(SVC(), parameters, cv=kfold, 
                     verbose=1, 
                     refit=True,    # 가장 좋은 모델 한번 더 돌리기 
                     n_jobs=-1,     # cpu 모든 코어 다 쓰기 (24개의 코어) 
                     n_iter=11,     # n 번 돌림 
                     random_state=3333 
                     )  

st = time.time()
model.fit(x_train, y_train)
et = time.time()

print('최적의 매개변수 :', model.best_estimator_)
# 최적의 매개변수 : SVC(C=1, kernel='linear')

print('최적의 파라미터 :', model.best_params_)
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score :', model.best_score_)    # train만 들어간 score, 훈련에서의 최고점
# best_score : 0.9833333333333334

print('model.score', model.score(x_test, y_test))   # test로 뽑은 score
# model.score 1.0

y_pre = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_pre))
# accuracy_score : 1.0

y_pre_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', accuracy_score(y_test, y_pre_best))      # 요걸로 확인하기
# 최적 튠 ACC : 1.0

print('걸린 시간 :', round(et-st, 2), '초')
# 걸린 시간 : 1.18 초

# Fitting 5 folds for each of 54 candidates, totalling 270 fits


# Random
# Fitting 5 folds for each of 10 candidates, totalling 50 fits  # 항상 k-fold x 10 번 돌아감
# 최적의 매개변수 : SVC(C=1000, kernel='linear')
# 최적의 파라미터 : {'kernel': 'linear', 'degree': 3, 'C': 1000}
# best_score : 0.9666666666666666
# model.score 1.0
# accuracy_score : 1.0
# 최적 튠 ACC : 1.0
# 걸린 시간 : 1.09 초