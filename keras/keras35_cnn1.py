from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D   # 2d 이미지 cnn

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))    # (4,4,10)
# (2,2) : 조각내는 픽셀 단위(커널사이즈) / (5,5,1) : (장수<-모름) x 가로 x 세로 x color(1or3)
# 10 : 증폭 단위
# 2x2로 잘랐을 때 shape => (4,4,10)
# 10 부분을 크게 하면 증폭, 10 대신 100 으로 하면 (4,4,100), 100장이 생성된다는 의미
# 다음 Dense 의 input shape 은 (4,4,10)
# 이미지는 압축되고 개수는 늘어남 -> 특성이 압축된 데이터가 증폭

model.add(Conv2D(5, (2,2)))     #(3,3,5)
model.summary()
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50        

#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0


