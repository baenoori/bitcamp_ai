import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
import time
from sklearn.utils import all_estimators

import sklearn as sk

### warning 무시 ###
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
iris = load_iris(return_X_y=True)
cancar = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancar, wine, digits]
data_name = ['아이리스', '켄서', '와인', '디지트']


#2. 모델 
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

st = time.time()

# for index, value in enumerate(datasets):        # enumerate : 데이터 + index 값도 같이 반환 (0,1,2,3), index : index, value : data
#     x, y = value
    
#     x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, 
#                                                     # stratify=y,
#                                                     )
    
#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)
    
#     for name, model in all:
#         try:                    # 에러 무시
#             #2. 모델
#             model = model()
            
#             #3. 훈련
#             model.fit(x_train, y_train)
#             scores = cross_val_score(model, x_train, y_train, cv=kfold)

            
#             # #4. 평가
#             # acc = model.score(x_test, y_test)
#             # print(name, '의 정답률 :', acc)
            
#             y_pre = cross_val_predict(model, x_test, y_test, cv=kfold)
#             print("============", data_name[index], name, "============")
#             print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 
#             acc = accuracy_score(y_test, y_pre)
#             print('cross_val_predict ACC :', acc)
#         except:
#             print("===================================")
#             print(name, '모델 오류') 

for index, value in enumerate(datasets):        # enumerate : 데이터 + index 값도 같이 반환 (0,1,2,3), index : index, value : data
    x, y = value
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, 
                                                    # stratify=y,
                                                    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print("===============", data_name[index],"===============")
    model_name=[]
    model_acc=[]
    
    for name, model in all:
        try:                    # 에러 무시
            #2. 모델
            model = model()
            
            #3. 훈련
            model.fit(x_train, y_train)
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            
            # #4. 평가
            acc = model.score(x_test, y_test)
            # print(name, '의 정답률 :', acc)
            
            y_pre = cross_val_predict(model, x_test, y_test, cv=kfold)
            # print("============", data_name[index], name, "============")
            # print('acc : ', scores, '\n평균 acc :', round(np.mean(scores), 4)) 
            acc = accuracy_score(y_test, y_pre)
            # print('cross_val_predict ACC :', acc)
            # model_info = np.array([name, acc])
            acc = round(np.mean(scores), 4)
            model_name.append(name)
            model_acc.append(acc)
        except:
            # print("===================================")
            # print(name, '모델 오류') 
            continue
    model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.where(model_acc == np.max(model_acc))
    print('최고 모델 :', model_name[max_index], 'ACC :', model_acc[max_index])

et = time.time()

print('걸린 시간 :', round(et-st), '초')

# 걸린 시간 : 86 초

# =============== 아이리스 ===============
# 최고 모델 : ['LinearDiscriminantAnalysis'] ACC : [0.9833]
# =============== 켄서 ===============
# 최고 모델 : ['LogisticRegressionCV'] ACC : [0.9758]
# =============== 와인 ===============
# 최고 모델 : ['CalibratedClassifierCV' 'RidgeClassifier' 'RidgeClassifierCV'] ACC : [0.9931 0.9931 0.9931]
# =============== 디지트 ===============
# 최고 모델 : ['ExtraTreesClassifier'] ACC : [0.984]

# 걸린 시간 : 88 초