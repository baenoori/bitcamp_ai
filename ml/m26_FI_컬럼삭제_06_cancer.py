from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

random_state = 1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, 
                                                    stratify=y
                                                    )

#2. 모델구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

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

# plt.subplot(2,2,1)
# plot_feature_importance_datasets(model1)

# plt.subplot(2,2,2)
# plot_feature_importance_datasets(model2)

# plt.subplot(2,2,3)
# plot_feature_importance_datasets(model3)

# plt.subplot(2,2,4)
# plot_feature_importance_datasets(model4)


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
x = x.drop(columns=col)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, 
                                                    stratify=y
                                                    )

model2.fit(x_train, y_train)
print('acc :', model2.score(x_test,y_test))

# acc : 0.9298245614035088