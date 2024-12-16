import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score

from bayes_opt import BayesianOptimization

import time

#1. 데이터
x, y = load_diabetes(return_X_y=True)

# y = pd.get_dummies(y)
# y = y.reshape(-1,1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=336, train_size=0.8, 
                                                    # stratify=y
                                                    )

print(x_train.shape, y_train.shape) # (353, 10) (353,)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lebal  = LabelEncoder()

optimizer_list = ['adam', 'rmsprop', 'adadelta']
activation_list = ['relu', 'elu', 'selu', 'linear']

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
# def build_model(drop, optimizer, activation,
#                 node1, node2, node3, node4, node5):
    activation = activation_list[int(activation)]
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer_list[int(optimizer)], metrics=['mae'], loss='mse')
    
    model.fit(x_train, y_train, epochs=100, 
            #   callbacks = [mcp, es, rlr],
              validation_split = 0.1,
            #   batch_size=batchs,
              verbose=0,
              )
    
    y_pre = model.predict(x_test)
    
    result = r2_score(y_test, y_pre)
    
    return result     


# print(optimizer)
# exit()
def create_hyperparameter():
    # batchs = (8, 64)
    # optimizers = ['adam', 'rmsprop', 'adadelta']
    
    # optimizers = ['adam', 'rmsprop', 'adadelta']
    # optimizers = (0, max(le.fit_transform(optimizers)))
    optimizers = (0, 2)
    dropouts = (0.2, 0.5)
    # activations = ['relu', 'elu', 'selu', 'linear']
    # activations = (0, max(lebal.fit_transform(activations)))
    activations = (0, 3)
    node1 = (16, 128)
    node2 = (16, 128)
    node3 = (16, 128)
    node4 = (16, 128)
    node5 = (16, 128)
    return {
        # 'batch_size' : batchs,
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
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1, 
                             )

bay = BayesianOptimization(
    f=build_model,
    pbounds=hyperparameters,
    random_state=333    
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # maximize 가 fit이라고 생각
et = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간 :', round(et-st, 2), '초')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# #################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH_SAVE = 'C:/ai5/_save/keras72/03/'

filename = '{epoch:04d}-{val_loss:.8f}.hdf5'
filepath = ''.join([PATH_SAVE, 'keras72_03_', date, "_", filename])
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


# # model.fit(x_train, y_train, epochs=100, callbacks = [mcp, es, rlr], validation_split = 0.1)
# et = time.time()

# print('걸린 시간 :', round(et-st, 2), '초')
# print('model.best_params_', model.best_params_)
# print('model.best_estimator_', model.best_estimator_)
# print('model.best_score_', model.best_score_)
# print('model.score', model.score(x_test, y_test))



# 걸린 시간 : 19.5 초
# model.best_params_ {'optimizer': 'rmsprop', 'node5': 64, 'node4': 32, 'node3': 64, 'node2': 128, 'node1': 32, 'drop': 0.2, 'batch_size': 400, 'activation': 'selu'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasRegressor object at 0x0000025E22A1F790>
# model.best_score_ -28485.987109375
# model.score -28114.302734375

# 걸린 시간 : 54.78 초
# model.best_params_ {'optimizer': 'rmsprop', 'node5': 16, 'node4': 32, 'node3': 128, 'node2': 64, 'node1': 32, 'drop': 0.2, 'batch_size': 100, 'activation': 'selu'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasRegressor object at 0x0000023B85476820>
# model.best_score_ -3069.732568359375
# model.score -2976.94140625

# 걸린 시간 : 105.95 초
# model.best_params_ {'optimizer': 'adam', 'node5': 16, 'node4': 64, 'node3': 32, 'node2': 128, 'node1': 64, 'drop': 0.3, 'batch_size': 1, 'activation': 'linear'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasRegressor object at 0x00000245491B60A0>
# model.best_score_ -3097.3343098958335
# 89/89 [==============================] - 0s 932us/step - loss: 3037.6052 - mae: 44.3351
# model.score -3037.605224609375

