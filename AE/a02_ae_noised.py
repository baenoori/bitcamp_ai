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

input_img = Input(shape=(28*28,))

##### 인코더
# encoded = Dense(64, activation='relu')(input_img) 
# encoded = Dense(1, activation='relu')(input_img)                # 데이터 소실
encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)

##### 디코더
# decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='linear')(encoded)      # 흐려짐 (초기 음수 부분 때문에)
decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)          # 흐려짐 (음수 때문에)

autoencoder = Model(input_img, decoded)

# 3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss ='mse')
# autoencoder.compile(optimizer='adam', loss ='binary_crossentropy')
autoencoder.fit(x_train_noised, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



