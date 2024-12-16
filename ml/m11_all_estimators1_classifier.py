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
                                                    stratify=y,
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

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
    

# AdaBoostClassifier 의 정답률 : 0.9666666666666667
# BaggingClassifier 의 정답률 : 0.9333333333333333
# BernoulliNB 의 정답률 : 0.6666666666666666
# CalibratedClassifierCV 의 정답률 : 0.9
# CategoricalNB 모델 오류
# ClassifierChain 모델 오류
# ComplementNB 모델 오류
# DecisionTreeClassifier 의 정답률 : 0.8333333333333334
# DummyClassifier 의 정답률 : 0.3333333333333333
# ExtraTreeClassifier 의 정답률 : 0.9666666666666667
# ExtraTreesClassifier 의 정답률 : 0.9
# GaussianNB 의 정답률 : 0.9666666666666667
# GaussianProcessClassifier 의 정답률 : 0.9666666666666667
# GradientBoostingClassifier 의 정답률 : 0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 : 0.9333333333333333
# KNeighborsClassifier 의 정답률 : 0.9
# LabelPropagation 의 정답률 : 0.9666666666666667
# LabelSpreading 의 정답률 : 0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률 : 1.0
# LinearSVC 의 정답률 : 0.9666666666666667
# LogisticRegression 의 정답률 : 0.9333333333333333
# LogisticRegressionCV 의 정답률 : 0.9
# MLPClassifier 의 정답률 : 0.9333333333333333
# MultiOutputClassifier 모델 오류
# MultinomialNB 모델 오류
# NearestCentroid 의 정답률 : 0.8
# NuSVC 의 정답률 : 0.9333333333333333
# OneVsOneClassifier 모델 오류
# OneVsRestClassifier 모델 오류
# OutputCodeClassifier 모델 오류
# PassiveAggressiveClassifier 의 정답률 : 1.0
# Perceptron 의 정답률 : 0.9333333333333333
# QuadraticDiscriminantAnalysis 의 정답률 : 0.9333333333333333
# RadiusNeighborsClassifier 의 정답률 : 0.8666666666666667
# RandomForestClassifier 의 정답률 : 0.9333333333333333
# RidgeClassifier 의 정답률 : 0.8666666666666667
# RidgeClassifierCV 의 정답률 : 0.8666666666666667
# SGDClassifier 의 정답률 : 0.8333333333333334
# SVC 의 정답률 : 0.9333333333333333
# StackingClassifier 모델 오류
# VotingClassifier 모델 오류


