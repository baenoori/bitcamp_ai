# 15개 행에서 5개를 더 넣어서 만들기 
# 어절 6개 짜리

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Conv1D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다',
    '집에 가고싶다', '오늘 점심 기대돼', '나는 오늘 밤에 잠을 많이 잘거야', '오늘 저녁 메뉴가 별로여서 먹지 않았다', '이번주 주말에는 부산으로 여행을 가고싶어 계획을 세웠다'    
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,1,1,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '오늘': 2, '너무': 3, '재미있다': 4, '최고에요': 5, '잘만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶어요': 15, '글쎄': 16, '별로에요': 
# 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '준영이': 25, '바보': 26, '반장': 27, '잘생겼다': 28, '태운이': 29, '또': 30, '구라친다': 31, '집 
# 에': 32, '가고싶다': 33, '점심': 34, '기대돼': 35, '나는': 36, '밤에': 37, '잠을': 38, '많이': 39, '잘거야': 40, '저녁': 41, '메뉴가': 42, '별로여서': 43, '먹지': 44, '않았다': 45, '이번주': 46, '주말에는': 47, '부
# 산으로': 48, '여행을': 49, '가고싶어': 50, '계획을': 51, '세웠다': 52}

x = token.texts_to_sequences(docs)
print(x)
# [[3, 4], [1, 5], [1, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [3, 23], 
# [1, 24], [25, 26], [27, 28], [29, 30, 31], [32, 33], [2, 34, 35], [36, 2, 37, 38, 39, 40], [2, 41, 42, 43, 44, 45], [46, 47, 48, 49, 50, 51, 52
print(type(x))  # <class 'list'>

### 패딩 ###
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                        # padding = 'pre', 'post',         # (디폴트) 앞에 0 / 뒤에 0 채우기
                        maxlen = 5,                      # n개로 자르기, 앞에가 잘림 
                        # truncating = 'pre', 'post,'      # 앞에서 / 뒤에서 부터 자르기                         
                         )
print(padded_x)
print(padded_x.shape)   # (20, 5)

padded_x = padded_x.reshape(20,5,1)

##ohe
from tensorflow.keras.utils import to_categorical
padded_x = to_categorical(padded_x)
print(padded_x.shape)  # (20, 5, 53)

x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, test_size=0.1, random_state=755)

#2. 모델 구성 
model = Sequential()
model.add(Conv1D(16, 3, input_shape=(5,53)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
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

path = './_save/keras65/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k65_03_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1, 
        #   validation_split=0.1,
        #   callbacks=[es] #, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

x_pre = ["태운이 참 재미없다"]
x_pre = token.texts_to_sequences(x_pre)
x_pre = pad_sequences(x_pre, maxlen = 5)
print(x_pre.shape)  # (1, 5)

x_pre = to_categorical(x_pre, num_classes=53)
print(x_pre.shape)  # (1, 5, 31)

y_pre = model.predict(x_pre)
print("태운이 참 재미없다 의 결과 :", np.round(y_pre))

x_pre2 = ["준영이는 또 지루해요"]
x_pre2 = token.texts_to_sequences(x_pre2)
x_pre2 = pad_sequences(x_pre2, maxlen=5)
x_pre2 = to_categorical(x_pre2, num_classes=53)
y_pre2 = model.predict(x_pre2)
print("준영이는 또 지루해요 의 결과 :", np.round(y_pre2))

x_pre3 = ["연기가 별로에요"]
x_pre3 = token.texts_to_sequences(x_pre3)
x_pre3 = pad_sequences(x_pre3, maxlen=5)
x_pre3 = to_categorical(x_pre3, num_classes=53)
y_pre3 = model.predict(x_pre3)
print("연기가 별로에요 의 결과 :", np.round(y_pre2))

print("걸린 시간 :", round(end-start,2),'초')


# loss : 4.043992042541504
# acc : 0.5
# 태운이 참 재미없다 의 결과 : [[1.]]
# 준영이는 또 지루해요 의 결과 : [[0.]]
# 연기가 별로에요 의 결과 : [[0.]]
# 걸린 시간 : 6.57 초
