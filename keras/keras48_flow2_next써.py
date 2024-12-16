from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

train_datagen =  ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=1,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

augment_size = 100      # 증가시킬 사이즈 

print(x_train.shape)    # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)
# x = x_train[0].reshape(28,28,1)

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)   # augment_size의 형태로 x_train[0]을 반복
print(aaa.shape)    # (100, 28, 28, 1)


xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),   # augment_size의 형태로 x_train[0]을 반복, x데이터
    np.zeros(augment_size),     # augment_size 형태의 0으로 가득찬 array 생성, y 데이터
    batch_size=augment_size,
    shuffle=False,
).next()

# xy_data : x,y 데이터가 같이있는 튜플 형태, x와 y 각 데이터는 numpy형태

print(xy_data)
print(type(xy_data))       # <class 'tuple'>

# print(xy_data.shape)     # AttributeError: 'tuple' object has no attribute 'shape'
print(len(xy_data))        # 2 (x와 y)
print(xy_data[0].shape)    # (100, 28, 28, 1) / x 데이터는 넘파이라 shape 찍힘
print(xy_data[1].shape)    # (100,)
# x데이터는 변환된 데이터 100개, y데이터는 다 0 -> y데이터는 상관 없고 x데이터를 변환 시켜줌

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)   # 7x7 판 하나 준비, i+1 => 인덱스 49개 생성 (index 1부터)
    plt.imshow(xy_data[0][i], cmap='gray')
    
plt.show()

