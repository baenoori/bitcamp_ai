import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ])

data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 0. 결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

# 1. 결측치 삭제
# print(data.dropna())          # 디폴트 : axis=0
# print(data.dropna(axis=0))    # 행 삭제
# print(data.dropna(axis=1))      # 열 삭제 

# 2-1. 특정값 - 평균
means = data.mean()
print(means)
data2 = data.fillna(means)      # 열 기준 평균 값으로 채워짐
print(data2)

# 2-2. 특정값 - 중위값
med = data.median()
print(means)
data3 = data.fillna(med)
print(data3)

# 2-3. 특정값 - 0
data4 = data.fillna(0)
print(data4)

data4_2 = data.fillna(777)
print(data4_2)

# 2-4. 특정값 - ffill (통상 마지막 값에)
data5 = data.ffill()
# data5 = data.fillna(method='ffill')
print(data5)                # # 가장 첫번쨰 행은 채울 값이 없어서 Nan 값으로 채워짐

# 2-5. 특정값 - bfill
data6 = data.bfill()        # 가장 마지막 행은 채울 값이 없어서 Nan 값으로 채워짐
# data6 = data.fillna(method='bfill')
print(data6)

####################### 특정 칼럼만 #######################
means = data['x1'].mean()
print(means)    # 6.5

med = data['x4'].median()
print(med)      # 6.0

data['x1'] = data['x1'].fillna(means)
data['x4'] = data['x4'].fillna(med)
data['x2'] = data['x2'].ffill()

print(data)
