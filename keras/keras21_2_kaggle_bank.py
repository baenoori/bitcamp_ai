# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:/ai5/_data/kaggle/playground-series-s4e1/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())

# 문자열 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']


print(x.shape)  # (165034, 10)
print(y.shape)  # (165034,)


# 정규화
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])
test_csv[:] = scalar.fit_transform(test_csv[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5324)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=90,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,
                 verbose=3,
                 validation_split=0.2,
                 callbacks=[es]
                 )

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)
print(y_pred)
y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')


### csv 파일 ###
y_submit = model.predict(test_csv)

# print(y_submit)
y_submit = np.round(y_submit)
# print(y_submit)
sampleSubmission_csv['Exited'] = y_submit
# print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path + "sampleSubmission_0723_1220.csv")

print(sampleSubmission_csv['Exited'].value_counts())


"""
epochs=10000, batch_size=100
test_size=0.1, random_state=5331
patience=60

loss : 0.1678706407546997
acc : 0.787
loss : 0.1678706407546997
acc : 0.787
r2 score : -5.1746370329119884e-05
acc_score : 0.786597188560349
걸린 시간 : 142.91 초


걸린 시간 : 85.12 초

binary_crossentropy 
loss : 0.32630449533462524
acc : 0.862

loss : 0.32269835472106934
acc : 0.86

"""
