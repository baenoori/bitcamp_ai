import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype("float32")/255.
x_test = x_test.reshape(10000, 28*28).astype("float32")/255.

                                        # 평균 0,표준편차가 0.1인 정규분포 형태의 랜덤값 
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)             # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))                       # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))         # 1.506013411202829 -0.5281790150375157

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)

# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

list = [1, 8, 32, 64, 154, 331, 486, 713]
outputs = []

outputs.append(x_test)
for i in list:
    print(f"============== {i}개 시작 ==============")
    model = autoencoder(hidden_layer_size=i)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)
    
    decoded_imgs = model.predict(x_test_noised)
    outputs.append(decoded_imgs)

import matplotlib.pyplot as plt
import random
fig, axes = plt.subplots(9, 5, figsize=(15, 15))

# 이미지 5개 랜덤
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
 