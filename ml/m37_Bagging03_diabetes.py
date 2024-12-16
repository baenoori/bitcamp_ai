import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression     # 분류 ! 
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier, BaggingRegressor

#1. 데이터 
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=4444,
                                                    # stratify=y
                                                    )

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
# model = BaggingRegressor(DecisionTreeRegressor(),
#                           n_estimators = 100,
#                           n_jobs = -1,
#                           random_state = 4444,
#                         #   bootstrap=True,       # 디폴트, 중복 허용
#                           bootstrap=False,       # 중복 허용 안함
                        #   )
# model = LogisticRegression()
# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators = 100,
#                           n_jobs = -1,
#                           random_state = 4444,
#                         #   bootstrap=True,       # 디폴트, 중복 허용
#                           bootstrap=False,       # 중복 허용 안함
#                           )
# model = RandomForestClassifier()
model = BaggingRegressor(RandomForestRegressor(),
                          n_estimators = 100,
                          n_jobs = -1,
                          random_state = 4444,
                          bootstrap=True,       # 디폴트, 중복 허용
                          # bootstrap=False,       # 중복 허용 안함
                          )
#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('최종 점수 :', result)

y_pre = model.predict(x_test)
acc = r2_score(y_test, y_pre)
print('acc :', acc)

# RandomForestRegressor - Bagging - bootstrap=False
# 최종 점수 : 0.49528747503870973
# acc : 0.49528747503870973

# RandomForestRegressor - Bagging - bootstrap=True
# 최종 점수 : 0.5277707117137402
# acc : 0.5277707117137402