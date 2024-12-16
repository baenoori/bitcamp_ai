#19 diabetes copy
# scaling 추가 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import time

#1. 데이터 
datesets = load_diabetes()
x = datesets.data
y = datesets.target

print(x, y) # (442, 10)
print(x.shape, y.shape) # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=555)

### scaling ###
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=2,
          verbose=3,           
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

print("걸린 시간 :",  round(end-start,2),'초')


# print("=================== hist ==================")
# print(hist)

# print("================ hist.history =============")
# print(hist.history)

# print("================ loss =============")
# print(hist.history['loss'])
# print("================ val_loss =============")
# print(hist.history['val_loss'])
# print("==================================================")

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'     # 한글 깨짐 해결, 폰트 적용

# plt.figure(figsize=(9,6))   # 9 x 6 사이즈 
# plt.plot(hist.history['loss'],c='red', label='loss',)  # y값 만 넣으면 시간 순으로 그려줌 
# plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')
# plt.legend(loc='upper right')   # 우측 상단 label 표시
# plt.title('디아벳 Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()  # 격자 표시
# plt.show()


""" 
test_size : 0.2
random_state : 722
epo : 500
batch_size : 2
loss : 2616.662109375
r2 score : 0.6206178868645644

validation_split=0.1
loss : 3425.0595703125
r2 score : 0.457290784457283

[걸린 시간 추가]_0719
epochs=1000
batch_size=2
loss : 3055.604736328125
r2 score : 0.5158318124802564
걸린 시간 : 55.6 초

[scaling 추가_0725 - minmax]
loss : 2959.938232421875
r2 score : 0.49198524450407544
걸린 시간 : 3.29 초

[scaling - StandardScaling]
loss : 3422.89599609375
r2 score : 0.45763361076733455
걸린 시간 : 1.72 초

[scaling - MaxAbsScaler]
loss : 2999.707763671875
r2 score : 0.5246888448164932
걸린 시간 : 0.88 초

[scaling - RobustScaler]
loss : 3207.403076171875
r2 score : 0.49177900289160625
걸린 시간 : 1.36 초

"""
