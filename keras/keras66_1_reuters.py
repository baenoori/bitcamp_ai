from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Bidirectional


(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 1000,     # 단어 사전의 개수,  빈도수의 개수가 높은 단어 순으로 1000개 뽑기
    # maxlen = 100,         # 단어가 100개까지 있는 문장만 뽑음 
    test_split=0.2,
)

print(x_train)      # numpy 안에 리스트 형태로 들어가 있음 -> 길이가 각각 다 다름, 조절 필요
print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)
print(y_train)  # [ 3  4  3 ... 25  3 25]
print(len(np.unique(y_train))) # 46
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train))    # <class 'numpy.ndarray'>
print(type(x_train[0]))    # <class 'list'>  <- numpy 로 바꿔주어야함
print(len(x_train[0]), len(x_train[1])) #  87 56 <- pad_sequence 필요

print("뉴스기사의 최대 길이 :", max(len(i) for i in x_train))   # 뉴스기사의 최대 길이 : 2376
print("뉴스기사의 최소 길이 :", min(len(i) for i in x_train))   # 뉴스기사의 최소 길이 : 13
print("뉴스기사의 평균 길이 :", sum(map(len, x_train))/len(x_train))   # 뉴스기사의 평균 길이 : 145.53

# 전처리
x_train = pad_sequences(x_train, padding='pre', maxlen=100, 
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, 
                        truncating='pre')

# y 원핫
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

model = Sequential()
model.add(Embedding(1000,100))
model.add(Bidirectional(LSTM(64)))     # (N, 10)
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras66/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k66_01_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(x_train, y_train, epochs=1000, batch_size=128, callbacks=[es, mcp], validation_split=0.1)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss/acc :', results)

# loss/acc : [1.347326636314392, 0.6736420392990112]