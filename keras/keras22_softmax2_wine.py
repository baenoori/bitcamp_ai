from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)      # (178, 13) (178,)
print(np.unique(y, return_counts=True))    # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

### one hot encoding ###
y = pd.get_dummies(y)
print(y)
print(y.shape)      # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=512,
                                                    stratify=y)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, batch_size=4,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es]
                 )

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],3))

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)
y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')


"""
random_state=2321
loss : 0.11503161489963531
acc : 0.944
r2 score : 0.9034274762382792
acc_score : 0.9444444444444444
걸린 시간 : 6.33 초

stratify=y
loss : 0.09611806273460388
acc : 0.944
r2 score : 0.9209355269931461
acc_score : 0.9444444444444444
걸린 시간 : 9.17 초

"""



