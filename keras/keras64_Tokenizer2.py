import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴롭힌다. 준영이는 못생겼다. 사영이는 마구 마구 더 못생겼다.'

# 맹들기
token = Tokenizer()     
token.fit_on_texts([text1, text2])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '못생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, '엄청': 9, 
# '먹었다': 10, '태운이는': 11, '선생을': 12, '괴롭힌다': 13, '준영이는': 14, '사영이는': 15, '더': 16}

print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 
# 1), ('마구', 6), ('먹었다', 1), ('태운이는', 1), ('선생을', 1), ('괴롭힌다', 1), ('준영이는', 1), ('못생
# 겼다', 2), ('사영이는', 1), ('더', 1)])

x = token.texts_to_sequences([text1, text2])    # 텍스트 수치화 
print(x)    
# [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

# x = sum(x, [])
x = np.concatenate(x)
print(x)
# [5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10, 11, 12, 13, 14, 4, 15, 1, 1, 16, 4]

### 원핫인코딩 ###
# 1. pandas
# x = pd.get_dummies(x)  # 2차원 리스트를 1차원으로 바꾸기
# print(x.shape)  # (24, 16)

# 2. sklearn
# from sklearn.preprocessing import OneHotEncoder
# x = np.reshape(x,(-1,1))
# ohe = OneHotEncoder(sparse=False)
# x = ohe.fit_transform(x)
# print(x.shape)  # (24, 16)

# 3. keras
from tensorflow.keras.utils import to_categorical
x = to_categorical(x)
x = x[:, 1:]
print(x.shape)  # (24, 16)



