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
    VGG16(include_top=False, input_shape=(100, 100, 3)),
    ResNet50(include_top=False, input_shape=(100, 100, 3)),
    ResNet101(include_top=False, input_shape=(100, 100, 3)),
    DenseNet121(include_top=False, input_shape=(100, 100, 3)),
    InceptionV3(include_top=False, input_shape=(100, 100, 3)),
    InceptionResNetV2(include_top=False, input_shape=(100, 100, 3)),
    MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
    # NASNetMobile(include_top=False, input_shape=(100, 100, 3)),
    EfficientNetB0(include_top=False, input_shape=(100, 100, 3)),
    Xception(include_top=False, input_shape=(100, 100, 3))
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
    model.add(Dense(10))
    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    np_path = "C:/ai5/_data/_save_npy/cat_dog_total/"
    x = np.load(np_path + 'keras49_05_x_train.npy')
    y = np.load(np_path + 'keras49_05_y_train.npy')
    xy_test = np.load(np_path + 'keras49_05_x_test.npy')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=921)

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
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    # print('loss :', loss[0])
    # print('acc :', round(loss[1],2))

    y_pre = model.predict(x_test)
    # print("걸린 시간 :", round(end-start,2),'초')
    print("-----------------------")
    print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', round(loss[1],2))
    
## 기존
# loss : 0.2779466509819031
# acc : 0.88833
# 걸린 시간 : 304.15 초

## 동결 X
# loss : 0.6932458877563477
# acc : 0.5
# 걸린 시간 : 1668.04 초

## 동결 O
# loss : 0.2594916522502899
# acc : 0.89

# Global average pooling
# loss : 0.6931847929954529
# acc : 0.5
# 걸린 시간 : 1969.96 초


