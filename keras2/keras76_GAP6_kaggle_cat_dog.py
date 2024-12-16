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

vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100, 3),
              )
vgg16.trainable = True     # 가중치 동결 

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

np_path = "C:/ai5/_data/_save_npy/cat_dog_total/"
x = np.load(np_path + 'keras49_05_x_train.npy')
y = np.load(np_path + 'keras49_05_y_train.npy')
xy_test = np.load(np_path + 'keras49_05_x_test.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=921)

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

##### 스케일링
# x_train = x_train/255.      # 0~1 사이 값으로 바뀜
# x_test = x_test/255.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=4,
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
print("걸린 시간 :", round(end-start,2),'초')

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

