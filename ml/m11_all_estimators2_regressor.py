import numpy as np
from sklearn.datasets import load_iris, load_boston
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
x,y = load_boston(return_X_y=True)
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

for name, model in all:
    try:                    # 에러 무시
        #2. 모델
        model = model()
        
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가
        acc = model.score(x_test, y_test)
        
        print(name, '의 정답률 :', acc)
    except:
        print(name, '모델 오류') 
    

# 모델의 개수 : 54
# ARDRegression 의 정답률 : 0.6490543159926918
# AdaBoostRegressor 의 정답률 : 0.803986216933891
# BaggingRegressor 의 정답률 : 0.7157847345462758
# BayesianRidge 의 정답률 : 0.6586373731345692
# CCA 의 정답률 : 0.5760637712936086
# DecisionTreeRegressor 의 정답률 : 0.45397863433144814
# DummyRegressor 의 정답률 : -0.007430492589588278
# ElasticNet 의 정답률 : 0.5731279253295222
# ElasticNetCV 의 정답률 : 0.6569708406332373
# ExtraTreeRegressor 의 정답률 : 0.5955633407173098
# ExtraTreesRegressor 의 정답률 : 0.8214997357521547
# GammaRegressor 의 정답률 : 0.6437876396947344
# GaussianProcessRegressor 의 정답률 : 0.1476543393562355
# GradientBoostingRegressor 의 정답률 : 0.8102553701631537
# HistGradientBoostingRegressor 의 정답률 : 0.7684401146184295
# HuberRegressor 의 정답률 : 0.6167220417229807
# IsotonicRegression 모델 오류
# KNeighborsRegressor 의 정답률 : 0.6881140273713647
# KernelRidge 의 정답률 : -5.440589335485418
# Lars 의 정답률 : 0.6592466510354096
# LarsCV 의 정답률 : 0.6564246721016527
# Lasso 의 정답률 : 0.5596073672171813
# LassoCV 의 정답률 : 0.6568838069466172
# LassoLars 의 정답률 : -0.007430492589588278
# LassoLarsCV 의 정답률 : 0.6564246721016527
# LassoLarsIC 의 정답률 : 0.6537996638134368
# LinearRegression 의 정답률 : 0.6592466510354096
# LinearSVR 의 정답률 : 0.6121199813095619
# MLPRegressor 의 정답률 : 0.6377561939696734
# MultiOutputRegressor 모델 오류
# MultiTaskElasticNet 모델 오류
# MultiTaskElasticNetCV 모델 오류
# MultiTaskLasso 모델 오류
# MultiTaskLassoCV 모델 오류
# NuSVR 의 정답률 : 0.571475022199101
# OrthogonalMatchingPursuit 의 정답률 : 0.4902618098232455
# OrthogonalMatchingPursuitCV 의 정답률 : 0.5895109685394244
# PLSCanonical 의 정답률 : -2.3425516644931874
# PLSRegression 의 정답률 : 0.6338953880621446
# PassiveAggressiveRegressor 의 정답률 : 0.4960426079370215
# PoissonRegressor 의 정답률 : 0.7223116318003031
# RANSACRegressor 의 정답률 : 0.48215662256775493
# RadiusNeighborsRegressor 모델 오류
# RandomForestRegressor 의 정답률 : 0.775781956800252
# RegressorChain 모델 오류
# Ridge 의 정답률 : 0.6591553638389476
# RidgeCV 의 정답률 : 0.6580694089156829
# SGDRegressor 의 정답률 : 0.6543224550532409
# SVR 의 정답률 : 0.5895829313193317
# StackingRegressor 모델 오류
# TheilSenRegressor 의 정답률 : 0.2308855258752336
# TransformedTargetRegressor 의 정답률 : 0.6592466510354096
# TweedieRegressor 의 정답률 : 0.5876808116324395
# VotingRegressor 모델 오류


