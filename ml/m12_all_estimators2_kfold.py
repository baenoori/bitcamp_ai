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

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

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
        print("============", name, "============")
        print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 
        acc = accuracy_score(y_test, y_pre)
        print('cross_val_predict ACC :', acc)
    except:
        print("========================")
        print(name, '모델 오류') 
        
        
#   모델의 개수 : 54
# ============ ARDRegression ============
# acc :  [0.63868074 0.80326571 0.64950009 0.75565513 0.80326603]
# 평균 acc : 0.7301
# ============ ARDRegression ============
# ARDRegression 모델 오류
# ============ AdaBoostRegressor ============
# acc :  [0.78436136 0.88647507 0.78356907 0.86361021 0.7894533 ]
# 평균 acc : 0.8215
# ============ AdaBoostRegressor ============
# AdaBoostRegressor 모델 오류
# ============ BaggingRegressor ============
# acc :  [0.86838965 0.87846945 0.75588992 0.84802466 0.85867693]
# 평균 acc : 0.8419
# ============ BaggingRegressor ============
# BaggingRegressor 모델 오류
# ============ BayesianRidge ============
# acc :  [0.66326568 0.80011305 0.64899875 0.73525937 0.80721508]
# 평균 acc : 0.731
# ============ BayesianRidge ============
# BayesianRidge 모델 오류
# ============ CCA ============
# acc :  [0.54281327 0.79572203 0.60751318 0.59169476 0.78076419]
# 평균 acc : 0.6637
# ============ CCA ============
# CCA 모델 오류
# ============ DecisionTreeRegressor ============
# acc :  [0.60452819 0.72822727 0.76504325 0.62302979 0.79247778]
# 평균 acc : 0.7027
# ============ DecisionTreeRegressor ============
# DecisionTreeRegressor 모델 오류
# ============ DummyRegressor ============
# acc :  [-0.02042588 -0.00380497 -0.0151419  -0.00379291 -0.00050612]
# 평균 acc : -0.0087
# ============ DummyRegressor ============
# DummyRegressor 모델 오류
# ============ ElasticNet ============
# acc :  [0.61378521 0.69820252 0.5481866  0.76382153 0.67612589]
# 평균 acc : 0.66
# ============ ElasticNet ============
# ElasticNet 모델 오류
# ============ ElasticNetCV ============
# acc :  [0.65967005 0.80116452 0.64714    0.73506048 0.79838304]
# 평균 acc : 0.7283
# ============ ElasticNetCV ============
# ElasticNetCV 모델 오류
# ============ ExtraTreeRegressor ============
# acc :  [0.78342412 0.76050588 0.64460939 0.560035   0.75942293]
# 평균 acc : 0.7016
# ============ ExtraTreeRegressor ============
# ExtraTreeRegressor 모델 오류
# ============ ExtraTreesRegressor ============
# acc :  [0.91718666 0.93702693 0.81490436 0.89140597 0.89161675]
# 평균 acc : 0.8904
# ============ ExtraTreesRegressor ============
# ExtraTreesRegressor 모델 오류
# ============ GammaRegressor ============
# acc :  [0.67999553 0.66333593 0.54781249 0.75488686 0.67312792]
# 평균 acc : 0.6638
# ============ GammaRegressor ============
# GammaRegressor 모델 오류
# ============ GaussianProcessRegressor ============
# acc :  [-0.10011669  0.18563898  0.14629962  0.2571788   0.68805235]
# 평균 acc : 0.2354
# ============ GaussianProcessRegressor ============
# GaussianProcessRegressor 모델 오류
# ============ GradientBoostingRegressor ============
# acc :  [0.89786149 0.89418794 0.8403309  0.87928204 0.88624293]
# 평균 acc : 0.8796
# ============ GradientBoostingRegressor ============
# GradientBoostingRegressor 모델 오류
# ============ HistGradientBoostingRegressor ============
# acc :  [0.85734264 0.90232283 0.78284771 0.89321474 0.87547845]
# 평균 acc : 0.8622
# ============ HistGradientBoostingRegressor ============
# HistGradientBoostingRegressor 모델 오류
# ============ HuberRegressor ============
# acc :  [0.67365455 0.78706976 0.62123757 0.75271374 0.77700355]
# 평균 acc : 0.7223
# ============ HuberRegressor ============
# HuberRegressor 모델 오류
# ============ IsotonicRegression ============
# IsotonicRegression 모델 오류
# ============ KNeighborsRegressor ============
# acc :  [0.77078147 0.78876115 0.58994399 0.74260466 0.8468651 ]
# 평균 acc : 0.7478
# ============ KNeighborsRegressor ============
# KNeighborsRegressor 모델 오류
# ============ KernelRidge ============
# acc :  [ -5.6576963   -4.70130817  -4.86509472 -10.52011856  -4.90005219]
# 평균 acc : -6.1289
# ============ KernelRidge ============
# KernelRidge 모델 오류
# ============ Lars ============
# acc :  [0.66230011 0.80378441 0.65206483 0.52031949 0.78899082]
# 평균 acc : 0.6855
# ============ Lars ============
# Lars 모델 오류
# ============ LarsCV ============
# acc :  [0.66230011 0.80406569 0.64929488 0.52449093 0.78943134]
# 평균 acc : 0.6859
# ============ LarsCV ============
# LarsCV 모델 오류
# ============ Lasso ============
# acc :  [0.59985393 0.75257776 0.59355763 0.78887393 0.70549672]
# 평균 acc : 0.6881
# ============ Lasso ============
# Lasso 모델 오류
# ============ LassoCV ============
# acc :  [0.65983166 0.80121149 0.65021793 0.72159501 0.80200915]
# 평균 acc : 0.727
# ============ LassoCV ============
# LassoCV 모델 오류
# ============ LassoLars ============
# acc :  [-0.02042588 -0.00380497 -0.0151419  -0.00379291 -0.00050612]
# 평균 acc : -0.0087
# ============ LassoLars ============
# LassoLars 모델 오류
# ============ LassoLarsCV ============
# acc :  [0.66230011 0.80147561 0.64929488 0.71901253 0.80257471]
# 평균 acc : 0.7269
# ============ LassoLarsCV ============
# LassoLarsCV 모델 오류
# ============ LassoLarsIC ============
# acc :  [0.60999431 0.80443606 0.64881422 0.76098796 0.77152429]
# 평균 acc : 0.7192
# ============ LassoLarsIC ============
# LassoLarsIC 모델 오류
# ============ LinearRegression ============
# acc :  [0.66230011 0.80082316 0.65206483 0.71662892 0.81009207]
# 평균 acc : 0.7284
# ============ LinearRegression ============
# LinearRegression 모델 오류
# ============ LinearSVR ============
# acc :  [0.67011858 0.78213434 0.6011667  0.75349701 0.76404459]
# 평균 acc : 0.7142
# ============ LinearSVR ============
# LinearSVR 모델 오류
# ============ MLPRegressor ============
# acc :  [0.61842169 0.65383716 0.58632883 0.58418034 0.75469497]
# 평균 acc : 0.6395
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
# acc :  [0.55582678 0.63630926 0.51789577 0.72089584 0.64504048]
# 평균 acc : 0.6152
# ============ NuSVR ============
# NuSVR 모델 오류
# ============ OrthogonalMatchingPursuit ============
# acc :  [0.5116866  0.62656178 0.46828951 0.57073488 0.57802883]
# 평균 acc : 0.5511
# ============ OrthogonalMatchingPursuit ============
# OrthogonalMatchingPursuit 모델 오류
# ============ OrthogonalMatchingPursuitCV ============
# acc :  [0.5898069  0.80025884 0.62552306 0.7782056  0.7785408 ]
# 평균 acc : 0.7145
# ============ OrthogonalMatchingPursuitCV ============
# OrthogonalMatchingPursuitCV 모델 오류
# ============ PLSCanonical ============
# acc :  [-2.7365765  -1.31014445 -1.57132275 -4.95566399 -1.8569158 ]
# 평균 acc : -2.4861
# ============ PLSCanonical ============
# PLSCanonical 모델 오류
# ============ PLSRegression ============
# acc :  [0.65166215 0.78598865 0.60848064 0.77088667 0.76807897]
# 평균 acc : 0.717
# ============ PLSRegression ============
# PLSRegression 모델 오류
# ============ PassiveAggressiveRegressor ============
# acc :  [0.50162864 0.7579294  0.39792342 0.43287933 0.53618595]
# 평균 acc : 0.5253
# ============ PassiveAggressiveRegressor ============
# PassiveAggressiveRegressor 모델 오류
# ============ PoissonRegressor ============
# acc :  [0.7661719  0.84488207 0.68690624 0.84053227 0.8464397 ]
# 평균 acc : 0.797
# ============ PoissonRegressor ============
# PoissonRegressor 모델 오류
# ============ RANSACRegressor ============
# acc :  [0.58036341 0.64958428 0.51338764 0.62181333 0.51073083]
# 평균 acc : 0.5752
# ============ RANSACRegressor ============
# RANSACRegressor 모델 오류
# ============ RadiusNeighborsRegressor ============
# RadiusNeighborsRegressor 모델 오류
# ============ RandomForestRegressor ============
# acc :  [0.85132704 0.88607367 0.79898706 0.89109699 0.88676903]
# 평균 acc : 0.8629
# ============ RandomForestRegressor ============
# RandomForestRegressor 모델 오류
# ============ RegressorChain ============
# RegressorChain 모델 오류
# ============ Ridge ============
# acc :  [0.66247992 0.80079859 0.651481   0.7207272  0.80972373] 
# 평균 acc : 0.729
# ============ Ridge ============
# Ridge 모델 오류
# ============ RidgeCV ============
# acc :  [0.66419007 0.79909662 0.64594369 0.7207272  0.80522799]
# 평균 acc : 0.727
# ============ RidgeCV ============
# RidgeCV 모델 오류
# ============ SGDRegressor ============
# acc :  [0.65809405 0.79773194 0.64815221 0.73772222 0.80687333]
# 평균 acc : 0.7297
# ============ SGDRegressor ============
# SGDRegressor 모델 오류
# ============ SVR ============
# acc :  [0.56825021 0.67660629 0.52924157 0.74515562 0.67486823]
# 평균 acc : 0.6388
# ============ SVR ============
# SVR 모델 오류
# ============ StackingRegressor ============
# StackingRegressor 모델 오류
# ============ TheilSenRegressor ============
# acc :  [0.51896991 0.6393724  0.47188235 0.63666374 0.69848667]
# 평균 acc : 0.5931
# ============ TheilSenRegressor ============
# TheilSenRegressor 모델 오류
# ============ TransformedTargetRegressor ============
# acc :  [0.66230011 0.80082316 0.65206483 0.71662892 0.81009207]
# 평균 acc : 0.7284
# ============ TransformedTargetRegressor ============
# TransformedTargetRegressor 모델 오류
# ============ TweedieRegressor ============
# acc :  [0.63627285 0.66491381 0.52444062 0.73619758 0.66673812]
# 평균 acc : 0.6457
# ============ TweedieRegressor ============
# TweedieRegressor 모델 오류
# ============ VotingRegressor ============
# VotingRegressor 모델 오류