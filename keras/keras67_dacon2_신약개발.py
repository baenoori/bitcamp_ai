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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Bidirectional, Embedding
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


# SMILES 데이터를 분자 지문으로 변환    추후 고도화 필요
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)    # MolFromSmiles : SMILES를 인풋으로 받아 해당 분자 구조를 저장하는 mol object를 반환
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])  #2048개를 분리
        return np.array(fp)     # 1값
    else:
        return np.zeros((CFG['NBITS'],))    # 0 값 


# 학습 ChEMBL 데이터 로드
path = 'C:/ai5/_data/dacon/신약개발/'
chembl_data = pd.read_csv(path + 'train.csv')  # 예시 파일 이름
chembl_data.head()
# chembl_data.shape   # (1952, 15)

train = chembl_data[['Smiles', 'pIC50']]                                # 모든 데이터를 다 쓸 수 있게 바꾸기 ~!~
train['Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint)     # 수치화 (0과 1로)

train_x = np.stack(train['Fingerprint'].values)     # 나머지 컬럼을 제외하고 fingerprint 컬럼만 train 에 사용
train_y = train['pIC50'].values

print(train_x.shape)        # (1952, 2048)      # 원핫 인코딩 된 값이라고 생각 할 수 있음 
print(train_x[0])

# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

print(train_x.shape)     # (1366, 2048)
print(train_y.shape)    # (1366,)

# 2. 모델 구성
model = Sequential()
model.add(Embedding(2048, 100))
model.add(LSTM(10))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path1 = './_save/keras67/기본1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path1, 'k67_01_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(train_x, train_y, epochs=1000, batch_size=128, callbacks=[es, mcp], validation_split=0.1)

end = time.time()


def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

# Validation 데이터로부터의 학습 모델 평가
val_y_pred = model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')
print('시간 : ', round(end - start), '초')

# RMSE: 2416.727459443392
# 시간 :  10.171529054641724

test = pd.read_csv(path + 'test.csv')
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)

test_y_pred = model.predict(test_x)

print(test_y_pred)

# submit = pd.read_csv(path + 'sample_submission.csv')
# submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
# submit.head()

# submit.to_csv('./baseline_submit.csv', index=False)

