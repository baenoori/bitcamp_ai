from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   # 이미지 땡겨오기
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'C:/ai5/_data/image/me/me.jpg'

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

# # me 폴더에 위에 데이터를 npy로 저장할 것
# np_path = "C:/ai5/_data/image/me/"
# np.save(np_path + 'keras46_me_arr_2.npy', arr=img)

######## 요기부터 증폭 ############

datagen =  ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    # horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

it = datagen.flow(img,                # 수치화된 데이터를 증폭
             batch_size=1,
            )                  
print(it)       # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001CB61B1AE80>

print(it.next())

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))   # 이미지를 5장 뽑음 (1행 5열짜리 이미지)
for i in range(5):
    batch = it.next()   # IDG에서 랜덤으로 한번 작업 (변환)
    print(batch.shape)  # (1, 100, 100, 3)
    batch = batch.reshape(100,100,3)
    
    ax[i].imshow(batch)
    ax[i].axis('off')
    
plt.show()
    



