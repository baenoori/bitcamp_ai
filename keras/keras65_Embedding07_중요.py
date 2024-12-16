import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Bidirectional
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

# padded_x = padded_x.reshape(15,5,1)

# ##ohe
# from tensorflow.keras.utils import to_categorical
# padded_x = to_categorical(padded_x)
# print(padded_x.shape)  # (15, 5, 31)


x_pre = ["태운이 참 재미없다"]
x_pre = token.texts_to_sequences(x_pre)
x_pre = pad_sequences(x_pre, maxlen = 5)
print(x_pre.shape)  # (1, 5)

# x_pre = to_categorical(x_pre, num_classes=31)
# print(x_pre.shape)  # (1, 5, 31)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

model = Sequential()
########### 임베딩 1 ###########
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))  # (N, 5, 100)
                    #input_dim : 단어사전의 개수(말뭉치의 개수)
                    #output_dim : 다음 레이어로 전달하는 노드의 개수 (조절 가능)
                    #input_length : (15,5), 전처리한 데이터의 컬럼의 개수
                    
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
# _________________________________________________________________
                    

########### 임베딩 2 ###########
# model.add(Embedding(input_dim=31, output_dim=100))  # input_length를 명시 안해줘도 알아서 조절해줌
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
# _________________________________________________________________

########### 임베딩 3 ###########
# model.add(Embedding(input_dim=100, output_dim=100))   # input_dim을 마음대로 조절은 가능하나 성능은 저하될 수 있음
                    #input_dim = 30 : 디폴트
                    #input_dim = 20 : 단어사전의 갯수보다 작을때 - 연산량이 줄어, 단어사전에서 임의로 뺌, 성능 저하
                    #input_dim = 40 : 단어사전의 갯수보다 클 때 - 임의의 랜덤 임베딩 생성, 성능 저하

########### 임베딩 4 ###########
# model.add(Embedding(31,100))    # 명칭 생략 가능, input_dim / output_dim  순서
# model.add(Embedding(31,100,5))    # ValueError: Could not interpret initializer identifier: 5 
# model.add(Embedding(31,100,input_length=5)) # input_length 는 명시 필요 
# model.add(Embedding(31,100,input_length=6))     # 수치 틀리면 안돌아감
model.add(Embedding(31,100,input_length=1))     # 1은 돌아감, input_length 의 약수는 가능함!!

model.add(LSTM(10))     # (N, 10)
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padded_x, labels, epochs=100)

#4. 평가, 예측
results = model.evaluate(padded_x, labels)
print('loss/acc :', results)
# loss/acc : [0.06594551354646683, 1.0]

y_pre = model.predict(x_pre)
print("태운이 참 재미없다 의 결과 :", np.round(y_pre))
# 태운이 참 재미없다 의 결과 : [[0.]]


