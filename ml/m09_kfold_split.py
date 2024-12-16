import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

#1. 데이터
datasets = load_iris()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

n_split = 3
kfold = KFold(n_splits=n_split, shuffle=False,
            #   random_state=333
              )

for train_index, val_index in kfold.split(df):
    print("==============")
    print(train_index, '\n', val_index)
    print('훈련데이터 개수 :', len(train_index), ' ', '검증데이터 개수: ', len(val_index))


