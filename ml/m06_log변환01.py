import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)
print(data)
print(data.shape)   # (1000,)
print(np.min(data), np.max(data))
# 0.002637017979792264 14.6721444423602

log_data = np.log1p(data)   # log 0 방지
# plt.subplot(1,2,1)
# plt.hist(data, bins=50, color='blue', alpha=0.5)
# plt.show()

## 로그 변환 데이터 히스토그램 그리지
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')
# plt.show()

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')
plt.show()


exp_data = np.expm1(log_data)   # 지수로 다시 변환
