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
    VGG16(include_top=False, input_shape=(32, 32, 3)),
    ResNet50(include_top=False, input_shape=(32, 32, 3)),
    ResNet101(include_top=False, input_shape=(32, 32, 3)),
    DenseNet121(include_top=False, input_shape=(32, 32, 3)),
    # InceptionV3(include_top=False, input_shape=(32, 32, 3)),
    # InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
    # NASNetMobile(include_top=False, input_shape=(32, 32, 3)),
    EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
    # Xception(include_top=False, input_shape=(32, 32, 3))
]
 
############ GAP 쓰기, 기존꺼 최고 꺼랑 성능 비교 ############

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import r2_score, accuracy_score

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

for model_i in model_list:
  model_i.trainable = False 
  
  model = Sequential()
  model.add(model_i)
  # model.add(Flatten())
  model.add(GlobalAveragePooling2D())
  model.add(Dense(100))
  model.add(Dense(100))
  model.add(Dense(10, activation='softmax'))

  # model.summary()

  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  #print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
  #print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

  ##### 스케일링
  x_train = x_train/255.      # 0~1 사이 값으로 바뀜
  x_test = x_test/255.

  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

  from tensorflow.keras.callbacks import EarlyStopping
  es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

  start = time.time()
  hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es],
            )
  end = time.time()

  #4. 평가, 예측
  loss = model.evaluate(x_test, y_test, verbose=1)
  
  print("-----------------------")
  print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', round(loss[1],2))
  
  # print('loss :', loss[0])
  # print('acc :', round(loss[1],2))

  y_pre = model.predict(x_test)

  y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
  y_test = np.argmax(y_test, axis=1).reshape(-1,1)

  # r2 = accuracy_score(y_test, y_pre)
  # print('accuracy_score :', r2)
  # print("걸린 시간 :", round(end-start,2),'초')


## 기존
# loss : 0.9801340103149414
# acc : 0.67

## 동결 X
# loss : 1.1994907855987549
# acc : 0.59
# accuracy_score : 0.099
# 걸린 시간 : 81.12 초

## 동결 O
# loss : 1.199491262435913
# acc : 0.59
# accuracy_score : 0.099
# 걸린 시간 : 80.51 초

# Global average pooling 사용
# loss : 2.3031632900238037
# acc : 0.1
# accuracy_score : 1.0
# 걸린 시간 : 124.77 초

# All transferLearning - 모델별
# 모델명 : vgg16 loss : 1.1986137628555298 acc : 0.59
# 모델명 : resnet50 loss : 1.637723445892334 acc : 0.42
# 모델명 : resnet101 loss : 1.8051543235778809 acc : 0.37
# 모델명 : densenet121 loss : 1.050866723060608 acc : 0.65
# 모델명 : mobilenetv2_1.00_224 loss : 1.9027105569839478 acc : 0.31
# 모델명 : efficientnetb0 loss : 2.303757905960083 acc : 0.1
