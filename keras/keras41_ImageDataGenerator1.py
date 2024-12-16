import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=1.2,              # 축소 또는 확대
    shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

test_datagen = ImageDataGenerator(
    rescale=1./255,              # test 데이터는 수치화만!! 
)

path_train = "./_data/image/brain/train/"
path_test = "./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
    path_train,                 # path_train 경로에 있는 모든 파일을 train_datagen 해줘라는 의미
    target_size=(200,200),      # 이미지 사이즈(규격) 일치 시키기 (resize), 큰거는 축소/작은거는 확대
    batch_size=10,              # 데이터가 80개면 10개씩 묶음 -> (10,200,200,1)이 8개 생김 
    class_mode='binary',        # 이진분류 - 두개의 클래스(0과 1, normal 과 add)
    color_mode='grayscale',     # 흑백
    shuffle=True, 
)
# 위 코드 실행 : Found 160 images belonging to 2 classes <- 160개의 데이터를 ad와 normal로 분류

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(200,200),
    batch_size=10,            
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,   # test 데이터는 shuffle할 필요 없음
)   
# Found 120 images belonging to 2 classes.

print(xy_train)
# # <keras.preprocessing.image.DirectoryIterator object at 0x000002BD64756700>
# print(xy_train.next())  # 이터레이터의 첫번쨰를 보여달라는 의미
# print(xy_train.next())  # 두번째 데이터 출력

print(xy_train[0])  # [1., 1., 0., 1., 1., 1., 0., 1., 1., 0.]
print(xy_train[0][0])    # 첫번쨰의 x데이터만 보기
print(xy_train[0][1])    # 첫번쨰의 y데이터만 보기 
# print(xy_train[0].shape)    # AttributeError: 'tuple' object has no attribute 'shape', tuple 의 0번쨰 - x, 1번쨰 - y데이터
print(xy_train[0][0].shape)   # (10, 200, 200, 1) , xy_train[0~15][0~1]
# print(xy_train[16])         # ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[15][2])        # IndexError: tuple index out of range

print(type(xy_train))         # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))      # <class 'tuple'>
print(type(xy_train[0][0]))   # <class 'numpy.ndarray'> - x
print(type(xy_train[0][1]))   # <class 'numpy.ndarray'> - y



