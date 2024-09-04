### 판다스로 바꿔서 컬럼 삭제
# pd.DataFrame
# 컬럼명 : datasets.feature_names 안에 있음

## 실습 ###
# 하위 20 ~ 25% 컬럼들 제거
# 데이터 재구성 후 기존 모델 결과와 비교 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

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

def plot_feature_importance_datasets(model):    # model : XGboost
    n_features = datasets.data.shape[1]     # 4
    plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

# plt.rc('xtick', labelsize=5)
# plt.rc('ytick', labelsize=5)
# plt.tight_layout()  # 간격 안겹치게 

plt.subplot(2,2,1)
plot_feature_importance_datasets(model1)

plt.subplot(2,2,2)
plot_feature_importance_datasets(model2)

plt.subplot(2,2,3)
plot_feature_importance_datasets(model3)

plt.subplot(2,2,4)
plot_feature_importance_datasets(model4)

# plt.show()

# ===================== DecisionTreeClassifier =====================
# acc : 1.0
# [0.01666667 0.         0.57742557 0.40590776]
# ===================== RandomForestClassifier =====================
# acc : 1.0
# [0.10691492 0.02814393 0.42049394 0.44444721]
# ===================== GradientBoostingClassifier =====================
# acc : 1.0
# [0.01074646 0.01084882 0.27282247 0.70558224]
# ===================== XGBClassifier =====================
# acc : 1.0
# [0.00897023 0.02282782 0.6855639  0.28263798]

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

### data 컬럼 삭제 ###
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
                                                    # stratify=y
                                                    )

# percent = np.percentile(model.feature_importances_, 25)

# rm_index=[]

# for index, importance in enumerate(model.feature_importances_):
#     if importance<=percent :
#         rm_index.append(index)

# x_train = np.delete(x_train, rm_index, axis=1)
# x_test = np.delete(x_test, rm_index, axis=1)


model2.fit(x_train, y_train)
print('acc :', model2.score(x_test,y_test))
# acc : 1.0