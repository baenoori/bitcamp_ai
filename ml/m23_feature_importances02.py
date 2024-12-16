from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
x,y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)      # (442, 10) (442,)

random_state = 1234

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, 
                                                    # stratify=y
                                                    )

#2. 모델구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

print("random_state :", random_state)
for model in models:
    model.fit(x_train, y_train)
    print("=====================", model.__class__.__name__, "=====================")
    print('r2 :', model.score(x_test,y_test))
    print(model.feature_importances_)

# random_state : 123
# ===================== DecisionTreeRegressor =====================
# acc : 0.15795709914946876
# [0.09559417 0.01904038 0.23114463 0.0534315  0.03604905 0.05879742
#  0.04902482 0.01682605 0.36525519 0.07483678]
# ===================== RandomForestRegressor =====================
# acc : 0.5265549614751442
# [0.05770917 0.01047587 0.28528549 0.09846103 0.04390962 0.05190847
#  0.05713042 0.02626033 0.28720491 0.08165469]
# ===================== GradientBoostingRegressor =====================
# acc : 0.5585049159804119
# [0.04935014 0.01077655 0.30278452 0.11174122 0.02686628 0.05718503
#  0.04058792 0.01773638 0.33840513 0.04456684]
# ===================== XGBRegressor =====================
# acc : 0.39065385219018145
# [0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
#  0.03822911 0.10475955 0.3368922  0.07076568]


# random_state : 1234
# ===================== DecisionTreeRegressor =====================
# acc : 0.0559650481222167
# [0.07158557 0.01303001 0.35155054 0.08699449 0.0324187  0.10286431
#  0.05514926 0.01286287 0.15727978 0.11626448]
# ===================== RandomForestRegressor =====================
# acc : 0.4284326658194092
# [0.05871688 0.01367855 0.31920498 0.08014276 0.04707416 0.05754524
#  0.05943644 0.02922561 0.2486196  0.08635578]
# ===================== GradientBoostingRegressor =====================
# acc : 0.41812130513779855
# [0.04638956 0.01552011 0.33619689 0.09551229 0.03163405 0.06614365
#  0.03816374 0.01418551 0.27744867 0.07880555]
# ===================== XGBRegressor =====================
# acc : 0.3141314179753749
# [0.02732131 0.08686274 0.27677354 0.07103474 0.04535247 0.07236516
#  0.0416071  0.11428294 0.17072852 0.09367147]
