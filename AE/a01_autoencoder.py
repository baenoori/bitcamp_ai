import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

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
# decoded = Dense(784, activation='sigmoid')(encoded)
decoded = Dense(784, activation='tanh')(encoded)          # 흐려짐 (음수 때문에)

autoencoder = Model(input_img, decoded)

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss ='mse')
autoencoder.compile(optimizer='adam', loss ='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# 인코더 5개의 히든과
# 디코더의 4개의 activation 
# 그리고 2개의 complie - loss 부분의 경우의 수
# 총 40가지의 로스를 정리하고
# 눈으로 결과치를 비교해볼것 / 뭐가 제일 좋은지 

