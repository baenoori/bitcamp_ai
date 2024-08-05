from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   # 이미지 땡겨오기
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


np_path = "C:/ai5/_data/image/me/"

test_me = np.load(np_path + 'keras46_me_arr.npy')

model2 = load_model('C:/ai5/_save/keras42_catdog/k42_0804_2154_0029-0.2737_0.28.hdf5')       

y_predict2 = model2.predict(test_me)
print(y_predict2)

# print('확률: ',1-y_predict2)


# [[1.]]
