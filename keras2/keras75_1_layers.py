import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, 
              input_shape=(32,32,3)
              )

vgg16.trainable = False     # 가중치 동결

model =  Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.trainable = False
model.summary()

print(len(model.weights))            # 30  : 13 (vgg16) * 2 (w, b) + 2(Dense) * 2 (w, b)
print(len(model.trainable_weights))  # 0

"""
                              |  trainable = True | model = False  | VGG = False
len(model.weights)            |  # 30             | # 30           | # 30
len(model.trainable_weights)  |  # 30             | # 0            | # 4

"""

