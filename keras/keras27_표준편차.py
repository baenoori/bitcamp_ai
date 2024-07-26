import numpy as np
from sklearn.preprocessing import StandardScaler

#1. 데이터
data = np.array([[1, 2, 3, 1],
                 [4, 5, 6, 2],
                 [7, 8, 9, 3],
                 [10,11,12,114],
                 [13,14,15,115]])

#1) 평균
means = np.mean(data, axis=0)
print('평균 :', means)  # 평균 : [ 7.  8.  9. 47.]

#2) 모집단 분산 (n으로 나누어줌)    # ddof=0 디폴트
population_variances = np.var(data, axis=0)
print('모집단 분산 :', population_variances)    # 모집단 분산 : [  18.   18.   18. 3038.]


#3) 표본 분산 (n-1으로 나누어줌)
variances = np.var(data, axis=0, ddof=1)    # ddof : n-1 하겠다는 의미
print('표본분산 :', variances)      # 표본분산 : [  22.5   22.5   22.5 3797.5]

#4) 표본 표준 편차
std = np.std(data, axis=0, ddof=1)
print("표본 표준편차 :", std)       # 표본 표준편차 : [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5) StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print('StandardScaler :\n', scaled_data)
# StandardScaler :
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]           #  모집단 분산으로 계산
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]
