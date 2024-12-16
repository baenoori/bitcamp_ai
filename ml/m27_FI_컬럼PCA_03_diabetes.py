from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

random_state = 1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, 
                                                    # stratify=y
                                                    )

#2. 모델구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

print("random_state :", random_state)
# for model in models:
#     model.fit(x_train, y_train)
#     print("=====================", model.__class__.__name__, "=====================")
#     print('acc :', model.score(x_test,y_test))
#     print(model.feature_importances_)

import matplotlib.pylab as plt
import numpy as np

# print(model)    # 파라미터 나옴

def plot_feature_importance_datasets(model):    # model : XGboost
    n_features = datasets.data.shape[1]     # 4
    plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

for i, model in enumerate(models):
    model.fit(x_train, y_train)
    print("=====================", model.__class__.__name__, "=====================")
    print('acc :', model.score(x_test,y_test))
    print(model.feature_importances_)
    plt.subplot(2,2,i+1)
    plot_feature_importance_datasets(model)

plt.rc('xtick', labelsize=5)
plt.rc('ytick', labelsize=5)
plt.tight_layout()  # 간격 안겹치게 
# plt.show()


import pandas as pd

percentile = np.array(np.percentile(model2.feature_importances_, 25))

col = []
for i, fi in enumerate(model2.feature_importances_):
    if fi <= percentile:
        col.append(datasets.feature_names[i])
    else:
        continue
print(col)

x = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x.drop(columns=col)
x2 = x[col]
print(x2)

print('x1: ', x1.shape) # x1:  (20640, 6)
print('x2 :',x2.shape)  # x2 : (20640, 2)

x_train1, x_test1, y_train, y_test = train_test_split(x1, y, train_size=0.8, random_state=random_state, 
                                                    # stratify=y
                                                    )

### PCA ###
x_train2, x_test2, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=random_state, 
                                                    # stratify=y
                                                    )
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x_train2 = pca.fit_transform(x_train2)
x_test2 = pca.transform(x_test2)

print(x_train1.shape)   # (353, 7)
print(x_train2.shape)   # (353, 1)
print(x_test1.shape)    # (89, 7)
print(x_test2.shape)    # (89, 1)

x_train = np.concatenate([x_train1, x_train2], axis=1)
x_test = np.concatenate([x_test1, x_test2], axis=1)

print(x_train.shape)    # (353, 8)
print(x_test.shape)     # (89, 8)

model2.fit(x_train, y_train)
print('acc :', model2.score(x_test,y_test))

# 열 삭제
# acc : 0.33406054338438773

#PCA
# acc : 0.3414858670900289