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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Bidirectional, Embedding, Conv1D, MaxPool1D
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

train_x = train_x.reshape(1952, 2048, 1)
# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

print(train_x.shape)     # (1366, 2048)
print(train_y.shape)    # (1366,)

# 2. 모델 구성
model = Sequential()
# model.add(Embedding(2048, 100))
model.add(Conv1D(512, 3, input_shape=(2048,1) ,activation='relu'))
model.add(MaxPool1D())

model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPool1D())

model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPool1D())

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path1 = './_save/keras67/3/'
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

model.fit(train_x, train_y, epochs=100, batch_size=64, callbacks=[es, mcp], validation_split=0.1)

end = time.time()


def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

# Validation 데이터로부터의 학습 모델 평가
val_y_pred = model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
rmse = np.sqrt(mse)

# print(f'RMSE: {rmse}')
# print('시간 : ', end - start)

test = pd.read_csv(path + 'test.csv')
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)
test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],1)
test_y_pred = model.predict(test_x)

# test_y_pred = pIC50_to_IC50(test_y_pred)

print(test_y_pred)
print(f'RMSE: {rmse}')
print('시간 : ', end - start)

submit = pd.read_csv(path + 'sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
submit.head()

submit.to_csv(path + 'submit_0820_1910.csv', index=False)



