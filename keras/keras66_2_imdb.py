from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Bidirectional
from sklearn.model_selection import train_test_split


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = 1000,     # 단어 사전의 개수,  빈도수의 개수가 높은 단어 순으로 1000개 뽑기
    # maxlen = 10,         # 단어가 100개까지 있는 문장만 뽑음 
    # test_split=0.2,
)

print(x_train) 
print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)
print(y_train)  # [1 0 0 ... 0 1 0]
print(np.unique(y_train, return_counts=True)) # 2
print(np.unique(y_test, return_counts=True))  # 2


print(type(x_train))        # <class 'numpy.ndarray'>
print(type(x_train[0]))     # <class 'list'>
print(len(x_train[0]), len(x_train[1]))     # 218 189

print("imdb의 최대 길이 :", max(len(i) for i in x_train))   # imdb의 최대 길이 : 2494
print("imdb의 최소 길이 :", min(len(i) for i in x_train))   # imdb의 최소 길이 : 11
print("imdb의 평균 길이 :", sum(map(len, x_train))/len(x_train))  # imdb의 평균 길이 : 238.71364

# 전처리
x_train = pad_sequences(x_train, padding='pre', maxlen=100, 
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, 
                        truncating='pre')

# ### 스케일링 
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

model = Sequential()
model.add(Embedding(1000,100))
model.add(Bidirectional(LSTM(64)))     # (N, 10)
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
filepath = "".join([path, 'k66_02_', date, '_', filename])   
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

# loss/acc : [0.3735301196575165, 0.8321999907493591]

# scaler 추가 
# loss/acc : [0.6932381987571716, 0.5]

