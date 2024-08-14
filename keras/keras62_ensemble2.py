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
x3_datasets = np.array([range(100),range(301,401),range(77,177), range(33,133)]).T      # (100,4)

y = np.array(range(3001, 3101))  # 한강의 화씨 온도 

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y, test_size=0.3, random_state=321)


#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu',  name='bit1')(input1)
dense2 = Dense(20, activation='relu',  name='bit2')(dense1)
dense3 = Dense(30, activation='relu',  name='bit3')(dense2)
dense4 = Dense(40, activation='relu',  name='bit4')(dense3)
output1 = Dense(50, activation='relu',  name='bit5')(dense4)

#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu',  name='bit11')(input11)
dense21 = Dense(200, activation='relu',  name='bit21')(dense11)
output11 = Dense(300, activation='relu',  name='bit31')(dense21)

#2-3. 모델
input21 = Input(shape=(4,))
dense22 = Dense(100, activation='relu',  name='bit22')(input21)
dense23 = Dense(200, activation='relu',  name='bit23')(dense22)
dense24 = Dense(300, activation='relu',  name='bit24')(dense23)
dense25 = Dense(400, activation='relu',  name='bit25')(dense24)
output22 = Dense(500, activation='relu',  name='bit26')(dense25)


#2-4. 모델 합치기
from keras.layers.merge import concatenate, Concatenate
# merge1 = concatenate([output1, output11], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output11, output22])

merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input11, input21], outputs=last_output)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=100, verbose=3,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ###### 
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras62/2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k62_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=3,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit([x1_train,x2_train,x3_train], y_train, epochs=1000, batch_size=2, 
          validation_split=0.1,
          callbacks=[es, mcp],
          verbose=1,
          )
end = time.time()


#4. 평가, 예측
result = model.evaluate([x1_test,x2_test,x3_test], y_test, batch_size=1)
print('loss :', result)

# y_pred = model.predict([x1_datasets[-5:], x2_datasets[-5:]])
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249, 255)]).T
x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T

y_pred = model.predict([x1_pred,x2_pred,x3_pred])

print('예측 결과:', y_pred)

# loss : 0.06662353873252869
# 예측 결과: [[3091.3452]
#  [3093.7444]
#  [3096.9485]
#  [3100.1858]
#  [3103.6978]
#  [3107.2046]]



