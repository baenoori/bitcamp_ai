import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import r2_score, accuracy_score

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(224, 224, 3),
              )
vgg16.trainable = True     # 가중치 동결 

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

# 기본
# Total params: 138,468,754
# Trainable params: 138,468,754
# Non-trainable params: 0

# GlobalAveragePooling
# Total params: 14,777,098
# Trainable params: 14,777,098
# Non-trainable params: 0