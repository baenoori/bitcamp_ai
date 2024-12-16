import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=True)

model.layers[20].trainable = False

model.summary()
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# model.layers[17].trainable = False
# Total params: 138,357,544
# Trainable params: 135,997,736
# Non-trainable params: 2,359,808

# model.layers[20].trainable = False
# Total params: 138,357,544
# Trainable params: 35,593,000
# Non-trainable params: 102,764,544


import pandas as pd
pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

