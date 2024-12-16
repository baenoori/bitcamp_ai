import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def leakyrelu(x):
    return np.maximum(0.01*x, x)

# leakyrelu = np.maximum(0.01*x, x)

y = leakyrelu(x)

plt.plot(x, y)
plt.grid()
plt.show()
