# swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def silu(x):
#      return x * (1 / (1 + np.exp(-x)))
#     # x * sigmoid
#     # 문제점 : ReLu 보다 계산량이 많아서 모델이 커질수록 부담스러워짐

silu = lambda x : x * (1 / (1 + np.exp(-x)))

y = silu(x)

plt.plot(x, y)
plt.grid()
plt.show()
