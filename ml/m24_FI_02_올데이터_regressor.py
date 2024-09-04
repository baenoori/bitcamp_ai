# california, diabetes

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
data1 = fetch_california_housing()
data2 = load_diabetes()

datasets = [data1, data2]
dataset_name = ['california', 'diabetes']

random_state = 123

#2. 모델구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

print("random_state :", random_state)

for i, data in enumerate(datasets):
    x = data.data
    y = data.target
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, 
                                                        # stratify=y
                                                        )
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    print("##########################", dataset_name[i], "##########################")
    for model in models:
        model.fit(x_train, y_train)
        print("========", model.__class__.__name__, "=======")
        print('r2 :', model.score(x_test,y_test))
        print(model.feature_importances_)
    i = i+1
    print("##########################################################################")
    print("")
    
# random_state : 123
# ########################## california ##########################
# ======== DecisionTreeRegressor =======
# r2 : 0.6000320873754088
# [0.51933732 0.04854233 0.04799864 0.02724695 0.03268622 0.13136699
#  0.0991418  0.09367973]
# ======== RandomForestRegressor =======
# r2 : 0.8121690217687418
# [0.52251872 0.05198119 0.04717624 0.02923746 0.03156002 0.13387452
#  0.0920824  0.09156945]
# ======== GradientBoostingRegressor =======
# r2 : 0.7978378408140232
# [0.59864938 0.03019079 0.02141916 0.00492639 0.00427861 0.12193384
#  0.10819286 0.11040897]
# ======== XGBRegressor =======
# r2 : 0.83707103301617
# [0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
#  0.0921493  0.10859864]
# ##########################################################################

# ########################## diabetes ##########################
# ======== DecisionTreeRegressor =======
# r2 : 0.15795709914946876
# [0.09559417 0.01904038 0.23114463 0.0534315  0.03604905 0.05879742
#  0.04902482 0.01682605 0.36525519 0.07483678]
# ======== RandomForestRegressor =======
# r2 : 0.5265549614751442
# [0.05770917 0.01047587 0.28528549 0.09846103 0.04390962 0.05190847
#  0.05713042 0.02626033 0.28720491 0.08165469]
# ======== GradientBoostingRegressor =======
# r2 : 0.5585049159804119
# [0.04935014 0.01077655 0.30278452 0.11174122 0.02686628 0.05718503
#  0.04058792 0.01773638 0.33840513 0.04456684]
# ======== XGBRegressor =======
# r2 : 0.39065385219018145
# [0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
#  0.03822911 0.10475955 0.3368922  0.07076568]
# ##########################################################################