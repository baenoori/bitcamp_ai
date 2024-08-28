import autokeras as ak
import tensorflow as tf

print(ak.__version__)   # 1.0.20
print(tf.__version__)   # 2.15.1

import time

#1. 데이터

(x_train, y_trian), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_trian.shape)     # (60000, 28, 28) (60000,)

#2. 모델
model = ak.ImageClassifier( # 이미지 분류 모델, 딥러닝 모델
    overwrite=False,        # False 가 디폴트
    max_trials=3,           # 3개의 모델
)

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_trian, epochs=10, validation_split=0.15)
end_time = time.time()

###### 최적의 모델 출력 #######
best_model = model.export_model()       # 3개의 모델 중 최적의 모델
print(best_model.summary())

###### 최적의 모델 저장 ######
path = 'C:/ai5/_save/autokeras/'
best_model.save(path + 'keras70_autokeras1.h5')

#4. 평가 예측
y_pre = model.predict(x_test)
result = model.evaluate(x_test, y_test)
print('model 결과 :', result)

y_pre2 = best_model.predict(x_test)
# result2 = best_model.evaluate(x_test, y_test)
# print('best_model 결과 :', result2)

print('걸린 시간 :', round(end_time - start_time, 2), '초')




