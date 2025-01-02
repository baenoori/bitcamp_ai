# keras45_7 ~ 등을 참고 해서 남자 여자 사진에 노이즈를 주고  (내 사진도 노이즈 추가)
# 오토 인코더로 피부 미백 훈련 가중치를 만든다
# 그 가중치로 내 사진을 예측해서 피부 미백 

import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

st = time.time()
np.random.seed(333)
tf.random.set_seed(333)

test_datagen = ImageDataGenerator(
    rescale=1./255,              
)

#1. 데이터
np_path = "C:/ai5/_data/_save_npy/gender/"
x_train = np.load(np_path + 'keras45_07_x_train.npy')

path_test = "C:/Users/baenoori/Pictures/snfl/"
x_test = test_datagen.flow_from_directory(path_test, target_size=(100,100),  
     batch_size=30000,          
     class_mode='binary',  
     color_mode='rgb',  
     shuffle=True, )[0][0]

# np_path = "C:/ai5/_data/image/me/"
# x_test = np.load(np_path + 'keras46_me_arr.npy')

                                        # 평균 0,표준편차가 0.1인 정규분포 형태의 랜덤값 
x_train_noised = x_train[:5000] + np.random.normal(0, 0.1, size=x_train[:5000].shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)             # (27167, 100, 100, 3) (1, 100, 100, 3)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)

# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3,3),input_shape=(100,100,3), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (3,3), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (3,3), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size, (3,3), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(3, (3,3), padding='same'))
    return model

# hidden_size = 713       # pca 1.0
# hidden_size = 486       # pca 0.999
# hidden_size = 331       # pca 0.99
hidden_size = 128       # pca 0.95

model = autoencoder(hidden_layer_size=hidden_size)    

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss ='mse')
# autoencoder.compile(optimizer='adam', loss ='binary_crossentropy')
model.fit(x_train_noised, x_train[:5000], epochs=100, batch_size=16, validation_split=0.2)

# 4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)


import matplotlib.pyplot as plt
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))


# 원본 이미지 맨위에 그리기 
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[0])
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[0])
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더기 츨략힌 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[0])
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

et = time.time()
print('걸린 시간 ;', round(et-st))



