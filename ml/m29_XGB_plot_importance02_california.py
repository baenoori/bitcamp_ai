from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
datasets = fetch_california_housing()
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
for model in models:
    model.fit(x_train, y_train)
    print("=====================", model.__class__.__name__, "=====================")
    print('acc :', model.score(x_test,y_test))
    print(model.feature_importances_)

import matplotlib.pylab as plt
import numpy as np

# # print(model)    # 파라미터 나옴

# def plot_feature_importance_datasets(model):    # model : XGboost
#     n_features = datasets.data.shape[1]     # 4
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
#     plt.yticks(np.arange(n_features), model.feature_importances_)
#     plt.xlabel("feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)

# for i, model in enumerate(models):
#     model.fit(x_train, y_train)
#     print("=====================", model.__class__.__name__, "=====================")
#     print('acc :', model.score(x_test,y_test))
#     print(model.feature_importances_)
#     plt.subplot(2,2,i+1)
#     plot_feature_importance_datasets(model)

# plt.rc('xtick', labelsize=5)
# plt.rc('ytick', labelsize=5)
# plt.tight_layout()  # 간격 안겹치게 
# plt.show()

# plt.subplot(2,2,1)
# plot_feature_importance_datasets(model1)

# plt.subplot(2,2,2)
# plot_feature_importance_datasets(model2)

# plt.subplot(2,2,3)
# plot_feature_importance_datasets(model3)

# plt.subplot(2,2,4)
# plot_feature_importance_datasets(model4)

from xgboost.plotting import plot_importance 
plot_importance(model)
plt.show()
