import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def selu(x, alpha):
    return (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

# selu = (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

y = selu(x, 2)

plt.plot(x, y)
plt.grid()
plt.show()
