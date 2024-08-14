import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model 
from keras.layers import Dense, Input, concatenate, Concatenate
import time 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터 
x_datasets = np.array([range(100), range(301,401)]).T      # (100,2)
                        # 삼성 종가, 하이닉스 종가 

y1 = np.array(range(3001, 3101))  # 한강의 화씨 온도 
y2 = np.array(range(13001,13101))   # 비트코인 가격 

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test \
    = train_test_split(x_datasets, y1, y2, test_size=0.3, random_state=321)


#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu',  name='bit1')(input1)
dense2 = Dense(20, activation='relu',  name='bit2')(dense1)
dense3 = Dense(30, activation='relu',  name='bit3')(dense2)
dense4 = Dense(40, activation='relu',  name='bit4')(dense3)
output1 = Dense(50, activation='relu',  name='bit5')(dense4)


# 2-5. 분기 1 
# dense51 = Dense(100, activation='relu',  name='bit51')(output1)
# dense52 = Dense(200, activation='relu',  name='bit52')(dense51)
output_1 = Dense(1, activation='relu',  name='output_1')(output1)

# 2-6. 분기 2 
# dense61 = Dense(100, activation='relu',  name='bit61')(output1)
# dense62 = Dense(200, activation='relu',  name='bit62')(dense61)
output_2 = Dense(1, activation='relu',  name='output_2')(output1)

model = Model(inputs=input1, outputs=[output_1, output_2])
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

path = './_save/keras62/4/'
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

model.fit(x1_train, [y1_train, y2_train], epochs=1000, batch_size=2, 
          validation_split=0.1,
          callbacks=[es, mcp],
          verbose=1,
          )
end = time.time()


#4. 평가, 예측
result = model.evaluate(x1_test, [y1_test, y2_test], batch_size=1)
print('loss :', result)

# y_pred = model.predict([x1_datasets[-5:], x2_datasets[-5:]])
x1_pred = np.array([range(100,106), range(400,406)]).T

y_pred = model.predict(x1_pred)

print('예측 결과:', y_pred)

# loss : [0.0048596481792628765, 0.004724512342363596, 0.00013513564772438258]
# 예측 결과: [array([[3090.942 ],
#        [3091.94  ],
#        [3092.9382],
#        [3093.936 ],
#        [3094.9346],
#        [3095.9329]], dtype=float32), array([[13058.085],
#        [13059.085],
#        [13060.085],
#        [13061.086],
#        [13062.087],
#        [13063.087]], dtype=float32)]



