import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
# model = VGG16()
### 디폴트 ###
# model = VGG16(weights='imagenet',
#               include_top=True,
#               input_shape=(224,224,3),
#               )
#############
# model.summary()

###################### VGG-16 기본 모델 ######################
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0
#  ...
#  predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

model = VGG16(# weights='imagenet',
              include_top=False,
#               input_shape=(224,224,3),
              input_shape=(100, 100 ,3),
              )
model.summary()

# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________

################ include_top=False ################
#1. FC layer 없어짐 (직접 아래에 fc layer 명시 해주면 됌)
#2. input_shape를 원하는 데이터 shape 로 맞추기 
