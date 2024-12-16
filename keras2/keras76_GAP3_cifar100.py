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
from tensorflow.keras.datasets import cifar100

vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32, 3),
              )
vgg16.trainable = True     # 가중치 동결 

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))

model.summary()

### 실습 ###
# 비교할거 
# 1. 이전의 본인이 한 최상의 겨로가
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 시간까지 비교 하기 

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

## 기존
# loss : 2.424513101577759
# acc : 0.38

## 동결 X
# loss : 4.60542631149292
# acc : 0.01
# accuracy_score : 0.0
# 걸린 시간 : 85.58 초

## 동결 O
# loss : 2.6408698558807373
# acc : 0.34
# accuracy_score : 0.0122
# 걸린 시간 : 67.8 초

# Global average pooling
# loss : 4.605428218841553
# acc : 0.01
# accuracy_score : 0.0
# 걸린 시간 : 89.21 초


