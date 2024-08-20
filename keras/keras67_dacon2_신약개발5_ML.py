import pandas as pd
import numpy as np
import os
import random

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Bidirectional, Embedding, Conv1D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

### 상수값 정의 및 seed 고정 ###
CFG = {
    'NBITS':2048,
    'SEED':42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)                        # numpy의 랜덤값 고정 
seed_everything(CFG['SEED']) # Seed 고정

#1. 데이터 

# # SMILES 데이터를 분자 지문으로 변환   
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)        # 분자 생성 
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])  #2048개를 분리
        return np.array(fp)     # 1값
    else:
        return np.zeros((CFG['NBITS'],))    # 0 값 

# 학습 ChEMBL 데이터 로드
path = 'C:/ai5/_data/dacon/신약개발/'
chembl_data = pd.read_csv(path + 'train.csv')  # 예시 파일 이름
# chembl_data.shape   # (1952, 15)

train = chembl_data[['Smiles', 'pIC50']]                            
train['Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint) 

train_x = np.stack(train['Fingerprint'].values) 
train_y = train['pIC50'].values

print(train_x.shape)        # (1952, 2048)
# print(train_x[0])

# train_x = train_x.reshape(1952, 2048, 1)
# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

print(train_x.shape)     # (1366, 2048)
print(train_y.shape)    # (1366,)

#2. 모델 
model = RandomForestRegressor(
    n_estimators=1000,       # 트리의 개수 
    max_features=1.0,       # 하위 집합의 크기, 회귀
    max_depth=None, 
    min_samples_split=4, 
    min_samples_leaf=2, 
    min_weight_fraction_leaf=0.0, 
    max_leaf_nodes=None,
    min_impurity_decrease=0.0, 
    bootstrap=True, 
    n_jobs=None, 
    random_state=CFG['SEED'])

model.fit(train_x, train_y)

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

# Validation 데이터로부터의 학습 모델 평가
val_y_pred = model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')

test = pd.read_csv(path + 'test.csv')
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)

test_y_pred = model.predict(test_x)


submit = pd.read_csv(path + 'sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
submit.head()

submit.to_csv(path + 'submit_0820_1930.csv', index=False)



print(f'RMSE: {rmse}')



# RMSE: 2405.09289631417