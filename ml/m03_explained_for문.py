# train test split 후 scaling 및 PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # 분류는 classifiaer, 회귀는 regress
from sklearn.decomposition import PCA
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

print(x)
print(x.shape)  # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y, shuffle=True)
# stratify : y의 라벨의 개수에 맞춰서 train test 비율을 맞춰줌 

### scaler, pca 이전에 하는게 좋음 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

### PCA  <- 비지도 학습 
for i in range(x.shape[1]): 
    pca = PCA(n_components=i+1)   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    #2. 모델
    model = RandomForestClassifier(random_state=1)

    #3. 훈련
    model.fit(x_train1, y_train)     # epo 디폴트 100

    #4. 평가
    results = model.score(x_test1, y_test)
    print(x_train1.shape, x_test1.shape)
    print('model.score :', results)    

    # (120, 3) (30, 3)
    # model.score : 0.9666666666666667

evr = pca.explained_variance_ratio_     # 설명가능한 변화율
print('evr :',evr)  # [0.72944865 0.22881648 0.03624461]

print('evr_sum :',sum(evr)) # 0.9945097426678959, <-- 1이 안됨 : 손실이 좀 있음 

evr_cumsum = np.cumsum(evr)     #누적합
print('evr_cumsum :', evr_cumsum)
# [0.72944865 0.95826513 0.99450974 1.        ]

## 시각화 ## 
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

# (120, 1) (30, 1)
# model.score : 0.9333333333333333
# (120, 2) (30, 2)
# model.score : 0.9
# (120, 3) (30, 3)
# model.score : 0.9666666666666667
# (120, 4) (30, 4)
# model.score : 1.0
# evr : [0.72944865 0.22881648 0.03624461 0.00549026]
# evr_sum : 1.0000000000000002
# evr_cumsum : [0.72944865 0.95826513 0.99450974 1.        ]

