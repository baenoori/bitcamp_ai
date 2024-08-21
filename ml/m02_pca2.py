# train test split 후 scaling 및 PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # 분류는 classifiaer, 회귀는 regress
from sklearn.decomposition import PCA

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
pca = PCA(n_components=3)   # 4개의 컬럼이 3개로 바뀜
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#2. 모델
model = RandomForestClassifier(random_state=1)

#3. 훈련
model.fit(x_train, y_train)     # epo 디폴트 100

#4. 평가
results = model.score(x_test, y_test)
print(x_train.shape, x_test.shape)
print('model.score :', results)    

# (120, 3) (30, 3)
# model.score : 0.9666666666666667