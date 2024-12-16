import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, 
              input_shape=(32,32,3)
              )
model =  Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#1. 전체 동결
# model.trainable = False

#2. 전체 동결
# for layer in model.layers:
#     layer.trainable = False

#3. 부분 동결
print(model.layers)
# [<keras.engine.functional.Functional object at 0x000002297E698DC0>, 
# <keras.layers.core.flatten.Flatten object at 0x000002297F684AC0>, 
# <keras.layers.core.dense.Dense object at 0x000002297F6A33A0>, 
# <keras.layers.core.dense.Dense object at 0x000002297F6BA4F0>]

# print(model.layers[0])  # <keras.engine.functional.Functional object at 0x0000024695F78BB0>     # vgg16
# print(model.layers[1])  # Flatten / [2] : Dense ...

model.layers[0].trainable = False

model.summary()

import pandas as pd
pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)