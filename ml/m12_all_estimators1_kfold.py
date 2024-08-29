import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators

import sklearn as sk

### warning 무시 ###
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x,y = load_iris(return_X_y=True)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, 
                                                    # stratify=y,
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

print('all algorithms :', all)
print('sk version :', sk.__version__)   # 1.5.1
print('모델의 개수 :', len(all))    
# 모델의 개수 : 43
# 모델의 개수 : 55

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

for name, model in all:
    try:                    # 에러 무시
        #2. 모델
        model = model()
        
        #3. 훈련
        model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)

        
        # #4. 평가
        # acc = model.score(x_test, y_test)
        # print(name, '의 정답률 :', acc)
        
        y_pre = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_pre)
        print("============", name, "============")
        print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 
        print('cross_val_predict ACC :', acc)
    except:
        print("============", name, "============")
        print(name, '모델 오류') 
        # print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 
        # print('cross_val_predict ACC :', acc)


# ============ ARDRegression ============
# ARDRegression 모델 오류
# ============ AdaBoostRegressor ============
# acc :  [0.875      0.95183168 0.93619213 0.95313172 0.82915506]
# 평균 acc : 0.9091
# cross_val_predict ACC : 0.9666666666666667
# ============ BaggingRegressor ============
# BaggingRegressor 모델 오류
# ============ BayesianRidge ============
# BayesianRidge 모델 오류
# ============ CCA ============
# CCA 모델 오류
# ============ DecisionTreeRegressor ============
# acc :  [0.875      0.93314763 0.93314763 1.         0.7994429 ]
# 평균 acc : 0.9081
# cross_val_predict ACC : 0.9333333333333333
# ============ DummyRegressor ============
# DummyRegressor 모델 오류
# ============ ElasticNet ============
# ElasticNet 모델 오류
# ============ ElasticNetCV ============
# ElasticNetCV 모델 오류
# ============ ExtraTreeRegressor ============
# acc :  [0.9375     0.93314763 0.93314763 1.         0.93314763]
# 평균 acc : 0.9474
# cross_val_predict ACC : 0.9333333333333333
# ============ ExtraTreesRegressor ============
# ExtraTreesRegressor 모델 오류
# ============ GammaRegressor ============
# GammaRegressor 모델 오류
# ============ GaussianProcessRegressor ============
# GaussianProcessRegressor 모델 오류
# ============ GradientBoostingRegressor ============
# GradientBoostingRegressor 모델 오류
# ============ HistGradientBoostingRegressor ============
# HistGradientBoostingRegressor 모델 오류
# ============ HuberRegressor ============
# HuberRegressor 모델 오류
# ============ IsotonicRegression ============
# IsotonicRegression 모델 오류
# ============ KNeighborsRegressor ============
# KNeighborsRegressor 모델 오류
# ============ KernelRidge ============
# KernelRidge 모델 오류
# ============ Lars ============
# Lars 모델 오류
# ============ LarsCV ============
# LarsCV 모델 오류
# ============ Lasso ============
# Lasso 모델 오류
# ============ LassoCV ============
# LassoCV 모델 오류
# ============ LassoLars ============
# LassoLars 모델 오류
# ============ LassoLarsCV ============
# LassoLarsCV 모델 오류
# ============ LassoLarsIC ============
# LassoLarsIC 모델 오류
# ============ LinearRegression ============
# LinearRegression 모델 오류
# ============ LinearSVR ============
# LinearSVR 모델 오류
# ============ MLPRegressor ============
# MLPRegressor 모델 오류
# ============ MultiOutputRegressor ============
# MultiOutputRegressor 모델 오류
# ============ MultiTaskElasticNet ============
# MultiTaskElasticNet 모델 오류
# ============ MultiTaskElasticNetCV ============
# MultiTaskElasticNetCV 모델 오류
# ============ MultiTaskLasso ============
# MultiTaskLasso 모델 오류
# ============ MultiTaskLassoCV ============
# MultiTaskLassoCV 모델 오류
# ============ NuSVR ============
# NuSVR 모델 오류
# ============ OrthogonalMatchingPursuit ============
# OrthogonalMatchingPursuit 모델 오류
# ============ OrthogonalMatchingPursuitCV ============
# OrthogonalMatchingPursuitCV 모델 오류
# ============ PLSCanonical ============
# PLSCanonical 모델 오류
# ============ PLSRegression ============
# PLSRegression 모델 오류
# ============ PassiveAggressiveRegressor ============
# PassiveAggressiveRegressor 모델 오류
# ============ PoissonRegressor ============
# PoissonRegressor 모델 오류
# ============ QuantileRegressor ============
# acc :  [ 0.         -0.00278552 -0.00278552 -0.00278552 -0.00278552]
# 평균 acc : -0.0022
# cross_val_predict ACC : 0.2
# ============ RANSACRegressor ============
# RANSACRegressor 모델 오류
# ============ RadiusNeighborsRegressor ============
# RadiusNeighborsRegressor 모델 오류
# ============ RandomForestRegressor ============
# RandomForestRegressor 모델 오류
# ============ RegressorChain ============
# RegressorChain 모델 오류
# ============ Ridge ============
# Ridge 모델 오류
# ============ RidgeCV ============
# RidgeCV 모델 오류
# ============ SGDRegressor ============
# SGDRegressor 모델 오류
# ============ SVR ============
# SVR 모델 오류
# ============ StackingRegressor ============
# StackingRegressor 모델 오류
# ============ TheilSenRegressor ============
# TheilSenRegressor 모델 오류
# ============ TransformedTargetRegressor ============
# TransformedTargetRegressor 모델 오류
# ============ TweedieRegressor ============
# TweedieRegressor 모델 오류
# ============ VotingRegressor ============
# VotingRegressor 모델 오류
