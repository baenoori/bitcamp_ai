import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score

from bayes_opt import BayesianOptimization

import time

#1. 데이터
# x, y = mnist.load_data()

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=336, train_size=0.8, 
#                                                     # stratify=y
#                                                     )

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)  


# ##### 스케일링 1-1
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train))     # 1.0, 0.0

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
# print(x_train.shape, x_test.shape) # (60000, 28*28) (60000,)


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lebal  = LabelEncoder()

# print(x_train.shape, y_train.shape) # (60000, 784) (60000, 10)
from tensorflow.keras.optimizers import Adam

#2. 모델
def build_model(drop=0.5, optimizer=Adam(0.001), activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    activation = lebal.inverse_transform([int(activation)])[0]

    inputs = Input(shape=(784,), name='inputs')
    x = Dense(int(node1), activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(int(node2), activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node3), activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node4), activation=activation, name='hidden4')(x)
    x = Dense(int(node5), activation=activation, name='hidden5')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=le.inverse_transform([int(optimizer)])[0], metrics=['acc'], loss='categorical_crossentropy')
    model.fit(x_train, y_train, epochs=100, 
            #   callbacks = [mcp, es, rlr],
              validation_split = 0.1,
            #   batch_size=batchs,
              verbose=0,
              )
    
    y_pre = model.predict(x_test)
    
    result = r2_score(y_test, y_pre)
    
    return result     

def create_hyperparameter():
    # batchs = (8, 64)
    optimizers = ['adam', 'rmsprop', 'adadelta']
    optimizers = (0, max(le.fit_transform(optimizers)))
    dropouts = (0.2, 0.5)
    activations = ['relu', 'elu', 'selu', 'linear']
    activations = (0, max(lebal.fit_transform(activations)))
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
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

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
#################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH_SAVE = 'C:/ai5/_save/keras71/14/'

filename = '{epoch:04d}-{val_loss:.8f}.hdf5'
filepath = ''.join([PATH_SAVE, 'keras71_14_', date, "_", filename])
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

# import time
# st = time.time()
# model.fit(x_train, y_train, epochs=80, callbacks = [mcp, es, rlr], validation_split = 0.1)
# et = time.time()

# print('걸린 시간 :', round(et-st, 2), '초')
# print('model.best_params_', model.best_params_)
# print('model.best_estimator_', model.best_estimator_)
# print('model.best_score_', model.best_score_)
# print('model.score', model.score(x_test, y_test))

# 걸린 시간 : 2759.26 초
# model.best_params_ {'optimizer': 'adam', 'node5': 128, 'node4': 64, 'node3': 128, 'node2': 16, 'node1': 32, 'drop': 0.3, 'batch_size': 32, 'activation': 'relu'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasRegressor object at 0x0000027CD3062E80>
# model.best_score_ -0.21417353550593057
# model.score -0.1842508763074875
