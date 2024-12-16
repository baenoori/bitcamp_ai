# 23_1 copy
# scaling 추가
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (200000, 200)
print(y.shape)  # (200000,)

print(pd.value_counts(y, sort=True))    # 이진 분류
# 0    179902
# 1     20098

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5233,
                                                    stratify=y)


####### scaling (데이터 전처리) #######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
model = Sequential()
model.add(Dense(512, input_dim=200, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True    
)
model.fit(x_train, y_train, epochs=5000, batch_size=5000,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test,y_test,verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)
r2 = r2_score(y_test,y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end-start, 2), '초')

### csv 파일 만들기 ###
y_submit = model.predict(test_csv)
print(y_submit)

y_submit = np.round(y_submit)
print(y_submit)

submission_csv['target'] = y_submit
submission_csv.to_csv(path + "sampleSubmission_0725_1730_RS.csv")

print(submission_csv['target'].value_counts())


"""
random_state=5233
patience=100
epochs=5000, batch_size=5000
loss : 0.24998871982097626
acc : 0.91
r2 score : 0.20396239264613947
acc_score : 0.90655
걸린 시간 : 335.13 초

[scaling 추가 - minmax]
loss : 0.23247575759887695
acc : 0.92
r2 score : 0.26345618791347103
acc_score : 0.9152
걸린 시간 : 220.27 초

[scaling - Standard]
loss : 0.24165880680084229
acc : 0.91
r2 score : 0.23505359996363395
acc_score : 0.91145
걸린 시간 : 209.45 초

[scaling - MaxAbsScaler]
loss : 0.23832131922245026
acc : 0.91
r2 score : 0.2398409456465671
acc_score : 0.9107
걸린 시간 : 205.05 초

[scaling - RobustScaler]
loss : 0.2408507615327835
acc : 0.91
r2 score : 0.23682737415352673
acc_score : 0.9125
걸린 시간 : 244.83 초

"""





