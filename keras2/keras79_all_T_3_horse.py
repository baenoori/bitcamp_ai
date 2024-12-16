from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import EfficientNetB0

model_list = [
    VGG16(include_top=False, input_shape=(200, 200, 3)),
    ResNet50(include_top=False, input_shape=(200, 200, 3)),
    ResNet101(include_top=False, input_shape=(200, 200, 3)),
    DenseNet121(include_top=False, input_shape=(200, 200, 3)),
    InceptionV3(include_top=False, input_shape=(200, 200, 3)),
    InceptionResNetV2(include_top=False, input_shape=(200, 200, 3)),
    MobileNetV2(include_top=False, input_shape=(200, 200, 3)),
    # NASNetMobile(include_top=False, input_shape=(200, 200, 3)),
    EfficientNetB0(include_top=False, input_shape=(200, 200, 3)),
    Xception(include_top=False, input_shape=(200, 200, 3))
]

############ GAP 쓰기, 기존꺼 최고 꺼랑 성능 비교 ############

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

for model_i in model_list:
    model_i.trainable = False 

    model = Sequential()
    model.add(model_i)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    ### 실습 ###
    # 비교할거 
    # 1. 이전의 본인이 한 최상의 겨로가
    # 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=True 
    # 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
    # 시간까지 비교 하기 

    np_path = "c:/ai5/_data/_save_npy/horse/"

    x_train = np.load(np_path + 'keras45_02_x_train.npy')
    y_train = np.load(np_path + 'keras45_02_y_train.npy')


    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=921)

    # print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
    # print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

    ##### 스케일링
    # x_train = x_train/255.      # 0~1 사이 값으로 바뀜
    # x_test = x_test/255.

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    # print('loss :', loss[0])
    # print('acc :', round(loss[1],2))

    y_pre = model.predict(x_test)
    # print("걸린 시간 :", round(end-start,2),'초')

    print("-----------------------")
    print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', round(loss[1],2))

## 기존
# loss : 0.030986253172159195
# acc : 0.9881

## 동결 X
# loss : 0.6982610821723938
# acc : 0.46
# 걸린 시간 : 52.11 초


## 동결 O
# loss : 0.00015890866052359343
# acc : 1.0
# 걸린 시간 : 25.16 초

# Global average pooling
# loss : 0.698575496673584
# acc : 0.46
# 걸린 시간 : 72.48 초

# -----------------------
# 모델명 : vgg16 loss : 0.0003751504700630903 acc : 1.0
# -----------------------
# 모델명 : resnet50 loss : 0.19807498157024384 acc : 0.94
# -----------------------
# 모델명 : resnet101 loss : 0.1384524255990982 acc : 0.94
# ----------------------- 
# 모델명 : densenet121 loss : 2.4213086362578906e-05 acc : 1.0
# -----------------------
# 모델명 : inception_v3 loss : 4.5849399612052366e-05 acc : 1.0
# -----------------------
# 모델명 : inception_resnet_v2 loss : 3.4836500617529964e-07 acc : 1.0
# -----------------------
# 모델명 : mobilenetv2_1.00_224 loss : 1.8748366983345477e-06 acc : 1.0
# -----------------------
# 모델명 : efficientnetb0 loss : 0.6969051957130432 acc : 0.46
# -----------------------
# 모델명 : xception loss : 4.8833103392098565e-06 acc : 1.0