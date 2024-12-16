# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/overview
# 다중분류

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/kaggle/otto-group-product-classification-challenge/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_cav = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])
# print(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape, y.shape)     # (61878, 93) (61878,)

# one hot encoder
y = pd.get_dummies(y)
print(y.shape)      # (61878, 9)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=755)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=93, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(9, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=10000, batch_size=20,
          verbose=3,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],4))

y_pre = model.predict(x_test)
r2 = r2_score(y_test,y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end-start, 2), '초')

### csv 파일 만들기 ###
y_submit = model.predict(test_csv)
# print(y_submit)

y_submit = np.round(y_submit)
# print(y_submit)

sampleSubmission_cav[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit

sampleSubmission_cav.to_csv(path + "sampleSubmission_0724_2100.csv")



"""
random_state=4232
loss : 0.5503734350204468
acc : 0.7883
r2 score : 0.6202412402488502
acc_score : 0.7514544279250162
걸린 시간 : 398.94 초

random_state=4332
patience=100
epochs=10000, batch_size=500
"""
