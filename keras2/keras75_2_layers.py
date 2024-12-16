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

model.trainable = False
model.summary()

print(len(model.weights))            # 30  : 13 (vgg16) * 2 (w, b) + 2(Dense) * 2 (w, b)
print(len(model.trainable_weights))  # 0

import pandas as pd
pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

    
#                                                           Layer Type Layer Name  Layer Trainable
# 0  <keras.engine.functional.Functional object at 0x00000273DF3AAF10>      vgg16            False
# 1   <keras.layers.core.flatten.Flatten object at 0x00000273DF3D4BE0>    flatten            False
# 2       <keras.layers.core.dense.Dense object at 0x00000273DF3F34C0>      dense            False
# 3       <keras.layers.core.dense.Dense object at 0x00000273DF4832B0>    dense_1            False