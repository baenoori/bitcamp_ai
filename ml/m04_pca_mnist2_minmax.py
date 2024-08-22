# scaler 추가

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier  # 분류는 classifiaer, 회귀는 regress


(x_train, _), (x_test, _) = mnist.load_data()   # y 데이터를 뽑지 않고 언더바 _ 로 자리만 남겨둠 
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)  # (70000, 28, 28)

##### [실습] #####
# PCA를 통해 0.95 이상인 n_components는 몇개?
# 0.95 이상
# 0.99 이상
# 0.999 이상
# 1.0 일 때 몇개 ? 
# 힌트 : argmax 와 cunsum 사용 

x = x.reshape(70000,28*28)
# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape)  # (70000, 784)

# Minmax scaler
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
x = x/255.
print(np.min(x), np.max(x))  # 0.0 1.0

### PCA  <- 비지도 학습 
pca = PCA(n_components=28*28)   # 4개의 컬럼이 3개로 바뀜
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

# print('0.95 이상 :', np.min(np.where(evr_cumsum>=0.95))+1)
# print('0.99 이상 :', np.min(np.where(evr_cumsum>=0.99))+1)
# print('0.999 이상 :', np.min(np.where(evr_cumsum>=0.999))+1)
print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum >= 1.0)+1)
 
# 0.95 이상 : 154
# 0.99 이상 : 331
# 0.999 이상 : 486
# 1.0 일 때 : 713
