import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer()     # 인스턴스(객체) = 클래스(), 인스턴스 생성 
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
# 먼저 많이 나오는 순서대로 출력, 인덱스 라벨링

print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])
# 4 5 2 2 3 처럼 수치화 필요 

x = token.texts_to_sequences([text])    # 텍스트 수치화 
print(x)
# [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]  # list는 shape 없음, len 으로 확인 (1,9)
# 이 수치화된 데이터를 그대로 모델에 사용할 수 없음 -> 원핫 인코딩 필요 

##### 원핫 3가지 만들기 #####
# 1부터 시작함 !
# 1. pandas
# x = pd.get_dummies(sum(x, []))  # 2차원 리스트를 1차원으로 바꾸기
# x = pd.get_dummies(np.array(x).reshape(-1, ))  # 2차원 리스트를 1차원으로 바꾸기
# print(x)  # (14, 9)

# 2. sklearn
# from sklearn.preprocessing import OneHotEncoder
# x = np.reshape(x,(-1,1))
# ohe = OneHotEncoder(sparse=False)
# x = ohe.fit_transform(x)
# print(x.shape)  # (14, 9)

# 3. keras
from tensorflow.keras.utils import to_categorical   # keras 이용
x = to_categorical(x, num_classes=10)
x = x[:, :, 1:]
x = x.reshape(14,9)
print(x.shape)  # (14, 9)

