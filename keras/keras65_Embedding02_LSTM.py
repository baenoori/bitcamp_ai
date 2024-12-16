import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM
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
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니
# 다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해
# 요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, 
# '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], 
# [7, 8, 9], [10, 11, 12, 13, 14], [15], 
# [16], [17, 18], [19, 20], 
# [21], [2, 22], [1, 23], 
# [24, 25], [26, 27], [28, 29, 30]]
print(type(x))  # <class 'list'>

### 패딩 ###
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                        # padding = 'pre', 'post',         # (디폴트) 앞에 0 / 뒤에 0 채우기
                        maxlen = 5,                      # n개로 자르기, 앞에가 잘림 
                        # truncating = 'pre', 'post,'      # 앞에서 / 뒤에서 부터 자르기                         
                         )
print(padded_x)
print(padded_x.shape)   # (15, 5)

padded_x = padded_x.reshape(15,5,1)

x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, test_size=0.1, random_state=755)

#2. 모델 구성 
##### DNN 으로 만들기 #####
model = Sequential()
model.add(LSTM(16, input_shape=(5,1)))
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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
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
x_pre = pad_sequences(x_pre, maxlen=5)
print(x_pre)
x_pre = x_pre.reshape(-1,5,1)

y_pre = model.predict(x_pre)
print("태운이 참 재미없다 의 결과 :", np.round(y_pre))

print("걸린 시간 :", round(end-start,2),'초')

# loss : 15.234529495239258
# acc : 0.0
# [[0 0 4 1 3]]
# 태운이 참 재미없다 의 결과 : [[1.]]
# 걸린 시간 : 49.37 초