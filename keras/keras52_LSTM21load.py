from tensorflow.keras.models import Sequential, load_model
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]              
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 80 맞추기

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)      # (13, 3, 1)



## 모델 로드 ##
# model2 = load_model('C:/ai5/_save/keras52/k52_2_0807_1717_0042-0.3836.hdf5')       # 79.3
model2 = load_model('C:/ai5/_save/keras52/k52_2_0807_1739_0035-1.2620.hdf5')       # 79.5
loss2 = model2.evaluate(x, y, verbose=0)
print('loss :', loss2)

y_predict2 = model2.predict(x_predict.reshape(1,3,1))
print(y_predict2)

