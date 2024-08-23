from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score
import tensorflow as tf
tf.random.set_seed(337) # seed 고정 (첫 가중치가 고정)
np.random.seed(337)

datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=337)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001   # 디폴트 값
learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=100,
          batch_size=32,          
          )

#4. 평가, 예측
print("=============1. 기본 출력==========")
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr : {0}, loss : {1}'.format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr : {0}, r2 : {1}'.format(learning_rate, r2))

# 0.001
# loss : 34.17979431152344
# r2 : 0.6359827801992104

# 0.01
# loss : 34.06443405151367
# r2 : 0.6372114306767375

# 0.0001
# loss : 604.579833984375
# r2 : -5.4388169079983

# 0.005
# loss : 34.32440948486328
# r2 : 0.6344426401542673

# 0.05
# loss : 33.68345642089844
# r2 : 0.6412688875548384

# 0.009
# loss : 33.88270568847656
# r2 : 0.6391468667462534

# 0.00098
# loss : 34.107418060302734
# r2 : 0.6367536215286909


############# [실습] ###################
# lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]



