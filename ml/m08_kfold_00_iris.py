import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
x,y = load_iris(return_X_y=True)
print(x)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)     # fit 제공됨
print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 
# acc :  [1.         0.86666667 1.         0.96666667 0.96666667]
# 평균 acc : 0.96

