import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model 
from keras.layers import Dense, Input, concatenate, Concatenate
import time 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터 
x1_datasets = np.array([range(100), range(301,401)]).T      # (100,2)
                        # 삼성 종가, 하이닉스 종가 
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()     # (100,3)
                        # 원유, 환율, 금시세 

y = np.array(range(3001, 3101))  # 한강의 화씨 온도 

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, test_size=0.3, random_state=321)

print(x1_train.shape, x2_train.shape, y_train.shape)    # (70, 2) (70, 3) (70,)


#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu',  name='bit1')(input1)
dense2 = Dense(20, activation='relu',  name='bit2')(dense1)
dense3 = Dense(30, activation='relu',  name='bit3')(dense2)
dense4 = Dense(40, activation='relu',  name='bit4')(dense3)
output1 = Dense(50, activation='relu',  name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs=output1)      # 할 필요 X 
# model1.summary()

#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu',  name='bit11')(input11)
dense21 = Dense(200, activation='relu',  name='bit21')(dense11)
output11 = Dense(300, activation='relu',  name='bit31')(dense21)
# model2 = Model(inputs=input11, outputs=output11)    # 할 필요 X

#2-3. 모델 합치기
from keras.layers.merge import concatenate, Concatenate
# merge1 = concatenate([output1, output11], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output11])

merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input11], outputs=last_output)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=3,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ###### 
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras62/1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k61_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=3,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit([x1_train,x2_train], y_train, epochs=1000, batch_size=2, 
          validation_split=0.1,
          callbacks=[es, mcp],
          verbose=1,
          )
end = time.time()


#4. 평가, 예측
result = model.evaluate([x1_test,x2_test], y_test, batch_size=1)
print('loss :', result)

# y_pred = model.predict([x1_datasets[-5:], x2_datasets[-5:]])
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249, 255)]).T

y_pred = model.predict([x1_pred,x2_pred])

print('예측 결과:', y_pred)
# print('시간 :', end-start)

# loss : 0.00011953512876061723
# 예측 결과: [[3101.4387]
#  [3103.929 ]
#  [3106.4192]
#  [3108.8958]
#  [3111.3765]
#  [3113.9653]]

