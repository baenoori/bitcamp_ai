import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
# x, y = mnist(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=336, train_size=0.8, 
                                                    # stratify=y
                                                    # )
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000, 10)
from tensorflow.keras.optimizers import Adam

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(28,28,1), name='inputs')
    x = Conv2D(node1, (3,3), activation=activation, name='hidden1', strides=1, padding='same')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(node2, (3,3), activation=activation, name='hidden2', strides=1, padding='same')(x)
    x = Dropout(drop)(x)
    x = Conv2D(node3, (3,3), activation=activation, name='hidden3', strides=1, padding='same')(x)
    x = Dropout(drop)(x)
    x = Conv2D(node4, (3,3), activation=activation, name='hidden4', strides=1, padding='same')(x)
    x = Conv2D(node5, (3,3), activation=activation, name='hidden5', strides=1, padding='same')(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [32, 16, 128, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,      
        }


hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': ([100, 200, 300, 400, 500],), 'optimizer': (['adam', 'rmsprop', 'adadelta'],), 'drop': ([0.2, 0.3, 0.4, 0.5],), 'activation': (['relu', 'elu', 'selu', 'linear'],), 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16], 'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8]}

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

keras_model = KerasClassifier(build_fn=build_model, verbose=1, 
                             )

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, 
                           n_iter=2,
                        #    n_jobs=-1,
                           verbose=1,
                           )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH_SAVE = 'C:/ai5/_save/keras71/14_CNN/'

filename = '{epoch:04d}-{val_loss:.8f}.hdf5'
filepath = ''.join([PATH_SAVE, 'keras71_14_CNN_', date, "_", filename])
#################### mcp 세이브 파일명 만들기 끝 ###################
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 8,
    verbose = 1,
    factor = 0.8
)

import time
st = time.time()
model.fit(x_train, y_train, epochs=50, callbacks = [mcp, es, rlr], validation_split = 0.1)
et = time.time()

print('걸린 시간 :', round(et-st, 2), '초')
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score_', model.best_score_)
print('model.score', model.score(x_test, y_test))


# 걸린 시간 : 2222.3 초
# model.best_params_ {'optimizer': 'adam', 'node5': 64, 'node4': 32, 'node3': 128, 'node2': 64, 'node1': 128, 'drop': 0.3, 'batch_size': 32, 'activation': 'relu'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001DC8CBBDD00>
# model.best_score_ 0.9832166830698649
# model.score 0.9886999726295471