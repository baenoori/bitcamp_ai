import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

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
path = 'C:/ai5/_data/kaggle/bike-sharing-demand/'   # 절대경로 , 파이썬에서 \\a는 '\a'로 취급 특수문자 쓸 때 주의

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 'datatime'열은 인덱스 취급, 데이터로 X
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [0, 0] < list (2개 이상은 리스트)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=1234,
                                                    # stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print("========================================")
print(x_train.shape, x_test.shape)  # torch.Size([9797, 8]) torch.Size([1089, 8])
print(y_train.shape, y_test.shape)  # torch.Size([9797]) torch.Size([1089])
print(type(x_train), type(y_train)) # <class 'torch.Tensor'> <class 'torch.Tensor'>

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
        # self.sigmoid = nn.Sigmoid()
        
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
        # x = self.sigmoid(x)
        return x

model = Model(8, 1).to(DEVICE)

######################################################################

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()     # 훈련모드, 디폴트
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()     # 기울기(gradient)값 계산까지, 역전파 시작
    optimizer.step()    # 가중치(w) 갱신, 역전파 끝
    
    return loss.item()


epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))        # verbose

print("=============================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()    # 평가모드, 역전파x, 가중치 갱신x, 기울기 계산 할수 있기도함, dropout batchnorm x
    
    with torch.no_grad():
        y_pre = model(x)
        loss2 = criterion(y, y_pre)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)

### 밑에 완성하기 ###
y_predict = model(x_test)
r2 = r2_score(y_test.cpu().numpy(), y_predict.detach().cpu().numpy())

print('r2_score :', r2)

# 최종 loss : 20687.2734375
# r2_score : 0.37019842863082886