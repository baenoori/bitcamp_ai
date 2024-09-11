import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time 

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=336, train_size=0.8, 
                                                    stratify=y
                                                    )

print(x_train.shape, y_train.shape) # (455, 30) (455,)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lebal  = LabelEncoder()

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    
    activation = lebal.inverse_transform([int(activation)])[0]
    inputs = Input(shape=(30,), name='inputs')
    x = Dense(int(node1), activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(int(node2), activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node3), activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node4), activation=activation, name='hidden4')(x)
    x = Dense(int(node5), activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='sigmoid', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=le.inverse_transform([int(optimizer)])[0], metrics=['acc'], loss='binary_crossentropy')
    
    
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

# keras_model = KerasClassifier(build_fn=build_model, verbose=1, 
#                              )

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


# {'target': 0.8080719113349915, 'params': {'activation': 3.0, 'drop': 0.2, 'node1': 89.03657565207007, 'node2': 62.251803849785254, 'node3': 22.288411113472538, 'node4': 39.40172681400444, 'node5': 34.90701299017195, 'optimizer': 2.0}}
# 100 번 걸린 시간 : 1010.76 초









from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH_SAVE = 'C:/ai5/_save/keras71/06/'

filename = '{epoch:04d}-{val_loss:.8f}.hdf5'
filepath = ''.join([PATH_SAVE, 'keras71_06_', date, "_", filename])
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
# model.fit(x_train, y_train, epochs=100, callbacks = [mcp, es, rlr], validation_split = 0.1)
# et = time.time()

# print('걸린 시간 :', round(et-st, 2), '초')
# print('model.best_params_', model.best_params_)
# print('model.best_estimator_', model.best_estimator_)
# print('model.best_score_', model.best_score_)
# print('model.score', model.score(x_test, y_test))

# 걸린 시간 : 278.0 초
# model.best_params_ {'optimizer': 'adam', 'node5': 32, 'node4': 128, 'node3': 16, 'node2': 32, 'node1': 16, 'drop': 0.3, 'batch_size': 16, 'activation': 'relu'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001D8E439CF70>
# model.best_score_ 0.7978244423866272
# model.score 0.8947368264198303
