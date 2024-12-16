from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   # 이미지 땡겨오기
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split



np_path = "C:/ai5/_data/image/me/"

x_test = np.load(np_path + 'keras46_me_arr.npy')

model2 = load_model('C:/ai5/_save/keras45_gender/k45_08_0805_1515_0018-0.1725.hdf5')       

y_predict2 = model2.predict(x_test)
print(y_predict2)

# [[0.]]

# print('확률: ',1-y_predict2)




