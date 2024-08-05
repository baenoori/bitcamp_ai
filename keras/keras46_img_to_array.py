from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   # 이미지 땡겨오기
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np

# path = 'C:/ai5/_data/image/me/me.jpg'
path = 'C:/ai5/_data/image/me/다운.jpg'

img = load_img(path, target_size=(100,100),)

print(img)          # <PIL.Image.Image image mode=RGB size=200x200 at 0x1FA625E26A0>
print(type(img))    # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()        # 내 사진 보기

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (1200, 1800, 3) -> (100, 100, 3)
print(type(arr))    # <class 'numpy.ndarray'>

### 4차원으로 바꿔주기 (차원증가) ###
# arr = arr.reshape(1,100,100,3)
img = np.expand_dims(arr, axis=0)   # 차원 증가
print(img.shape)    # (1, 100, 100, 3)

# me 폴더에 위에 데이터를 npy로 저장할 것
np_path = "C:/ai5/_data/image/me/"
np.save(np_path + 'keras46_me_arr_8.npy', arr=img)
