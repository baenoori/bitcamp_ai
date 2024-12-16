import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

SEED = 0
import random
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch cuda 시드 고정

#1. 데이터
path = "C:/ai5/_data/dacon/diabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=369,
                                                    stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print("========================================")
print(x_train.shape, x_test.shape)  # torch.Size([586, 8]) torch.Size([66, 8])
print(y_train.shape, y_test.shape)  # torch.Size([586, 1]) torch.Size([66, 1])
print(type(x_train), type(y_train)) # <class 'torch.Tensor'> <class 'torch.Tensor'>

#################################### torch 데이터셋 만들기 ####################################
from torch.utils.data import TensorDataset  # x, y 합치기
from torch.utils.data import DataLoader     # batch 정의 

# 1. x와 y를 합친다
train_set = TensorDataset(x_train, y_train)     # tuple 형태로
test_set = TensorDataset(x_test, y_test)

#2. batch
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)    # TEST셋은 shuffle X

#################################### class로 모델 구성 ####################################

#2. 모델 구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()      # nn.Module에 있는 걸 모두 상속 받아서 쓰겠다는 의미 (디폴트)
        super(Model, self).__init__()       # nn.Module에 있는 Model과 self를 쓰겠다는 의미
        ### 정의 ###
        self.linear1 = nn.Linear(input_dim, 64)     # model 정의 
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    # 순전파!!! (위에 정의 해놓은 모델을 가지고 실행)
    def forward(self, input_size):              # nn.Module에 있는거  # method 
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x
                
model = Model(8, 1).to(DEVICE)

######################################################################

#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()     # 훈련모드, 디폴트
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()     # 기울기(gradient)값 계산까지, 역전파 시작
        optimizer.step()    # 가중치(w) 갱신, 역전파 끝
        total_loss += loss.item()
    return total_loss / len(loader)


epochs = 200
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss: {}'.format(epoch, loss))        # verbose


print("=============================")

#4. 평가, 예측
# loss = model.evalutate(x,y)
def evaluate(model, criterion, loader):
    model.eval()    # 평가모드, 역전파x, 가중치 갱신x, 기울기 계산 할수 있기도함, dropout batchnorm x
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad(): 
            y_pre = model(x_batch)
            loss2 = criterion(y_batch, y_pre)
            total_loss += loss2.item()
    return total_loss / len(loader)


last_loss = evaluate(model, criterion, test_loader)
print('최종 loss :', last_loss)

### 밑에 완성하기 ###
from sklearn.metrics import accuracy_score
y_predict = model(x_test)
acc = accuracy_score(y_test.cpu().numpy(), np.round(y_predict.detach().cpu().numpy()))

print('acc_score :', acc)

def acc_score(model, loader):
    x_test = []
    y_test = []
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
    x_test = torch.FloatTensor(x_test).to(DEVICE)
    y_pre = model(x_test)
    acc = accuracy_score(y_test, np.round(y_pre.detach().cpu().numpy()))
    return acc

acc = acc_score(model, test_loader)
print('acc_score :', acc)

# 최종 loss : 18.96337701938628
# acc_score : 0.6060606060606061
# acc_score : 0.6060606060606061