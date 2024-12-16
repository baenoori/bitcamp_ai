from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
# x,y = load_iris(return_X_y=True)
# print(x.shape, y.shape)      # (150, 4) (150,)
datasets = load_iris()
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
for model in models:
    model.fit(x_train, y_train)
    print("=====================", model.__class__.__name__, "=====================")
    print('acc :', model.score(x_test,y_test))
    print(model.feature_importances_)

import matplotlib.pylab as plt
import numpy as np

# print(model)    # 파라미터 나옴

# def plot_feature_importance_datasets(model):    # model : XGboost
#     n_features = datasets.data.shape[1]     # 4
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
#     plt.yticks(np.arange(n_features), model.feature_importances_)   # 눈금, 숫자 레이블 표시 
#     plt.xlabel("feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)        # 축 범위 설정 
#     plt.title(model.__class__.__name__)

# plot_feature_importance_datasets(model)
# plt.show()

from xgboost.plotting import plot_importance 
plot_importance(model)
plt.show()



