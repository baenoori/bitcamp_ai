# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y는 T (degC) 로 잡기, 자르는거는 마음대로 ~ (y :144개~)
# 맞추기 : 2016년 12월 31일 00시 10분부터 2017.01.01 00:00:00 까지 데이터 144개 (훈련에 쓰지 않음 )

import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
path1 = "C:/ai5/_data/kaggle/jena/"
datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv", index_col=0)

print(datasets.shape)   # (420551, 14)

a = datasets[:-144]
print(a.shape)      # (420407, 14)

y_cor = datasets[-144:]['T (degC)']            # 예측치 정답
print(y_cor.shape)    # (144,)

x_predict = datasets[-144:]
print(x_predict.shape)      # (144, 14)

size = 864                    # 144 * 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)        # (419544, 864, 14)

x = bbb[:, : -144, ]
y = bbb[:, -144 : , 1]       # T 데이터 

print(x.shape, y.shape) # (419544, 720, 14) (419544, 144)
# print(y)

