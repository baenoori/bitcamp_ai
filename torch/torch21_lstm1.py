import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)  # torch 고정, cpu 고정 
torch.cuda.manual_seed(333) # gpu 고정

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],           
             ])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)     # (7, 3) (7,)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)      # (7, 3, 1)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
print(x.shape, y.size())    # torch.Size([7, 3, 1]) torch.Size([7])

from torch.utils.data import TensorDataset  # x, y 합치기
from torch.utils.data import DataLoader     # batch 정의

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# aaa = iter(train_loader)
# bbb = next(aaa)
# print(bbb)

#2. 모델
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTM(input_size=1,       # input_size : feature개수, 나머지 사이즈는 알아서 맞춤 ~! 
                           hidden_size = 32,   # hidden_size : output 노드의 개수 
                           num_layers=1,       # 디폴트, 은닉층의 레이어 개수 (3 or 5가 좋다는 말이 있음) 
                           batch_first=True,   # 미지정 시 (2,3,1) 이 아닌 (3,2,1) 로 나옴, (2,3,1)로 나오게 해주세요~ 하는거임 (batch, Timestep, Feature)형식으로 나오데, False가 디폴트
                        #    bidirectional=True, # 디폴트 False
                           )    # (3, N, 1) -> batch_first=True -> (N, 3, 1) -> hidden layer -> (N, 3, 32)
        # self.cell = nn.RNN(1, 32, batch_first=True)
        self.fc1 = nn.Linear(3*32, 16)  # (N, 3*32) -> (N, 16) / reshape 부분은 forward 에서 정의
        self.fc2 = nn.Linear(16, 8)     # (N, 16) -> (N, 8)
        self.fc3 = nn.Linear(8, 1)      # (N, 8) -> (N, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, h0, c0):
        # model.add(SimpleRNN(32, input_shape=(3,1)))과 동일 
        #if h0 == None:
        h0 = torch.zeros(1, x.size(0), 32).to(DEVICE)  # (num_layers, batch_size, hidden_size), x와의 행렬 연산으로 위해 shape를 맞춰줌 
        c0 = torch.zeros(1, x.size(0), 32).to(DEVICE)  # (num_layers, batch_size, hidden_size), x와의 행렬 연산으로 위해 shape를 맞춰줌 
        
        # x, hidden_state = self.cell(x, h0)  # h0를 사실상 우리는 쓰지 않음
        x, (hn, cn) = self.cell(x, (h0, c0))  # h0를 사실상 우리는 쓰지 않음
        # x, _ = self.cell(x)
        x = self.relu(x)
        
        # x = x.reshape(-1, 3*32)
        # x = x.view(-1, 3*32)    # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        x = x.contiguous()
        x = x.view(-1, 3*32)  # *2 안해주면 biditrctonal 인식 x
        x = self.fc1(x)
    
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x) 
        
        return x
 
model = LSTM().to(DEVICE)

from torchsummary import summary
# summary(model, (3, 1))

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    model.train()
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)
        optimizer.zero_grad()
        
        h0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE) 
        c0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE) 
        hypothesis = model(x_batch, h0, c0)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()     # 기울기 계산
        optimizer.step()    # 가중치 갱신 
        
        epoch_loss += loss.item() 
    return epoch_loss / len(loader) 

def evalutate(model, criterion, loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            h0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE) 
            c0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE) 
            hypothesis = model(x_batch, h0, c0)
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
        return epoch_loss / len(loader)
    

for epoch in range(1, 1001):
    loss = train(model, criterion, optimizer, train_loader)
    
    if epoch % 20 == 0:
        print('epoch : {}, loss : {}'.format(epoch, loss))

x_predict = np.array([[8,9,10]])

def predict(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(2).to(DEVICE)  # (1,3) -> (1,3,1)
        
        h0 = torch.zeros(1, data.size(0), 32).to(DEVICE) 
        c0 = torch.zeros(1, data.size(0), 32).to(DEVICE) 
        y_predict = model(data, h0, c0)
    return y_predict.cpu().numpy()

y_pre = predict(model, x_predict)
print("================")
print(y_pre[0][0])


print(f"{x_predict} 의 예측값 : {y_pre[0][0]}")     #[[ 8  9 10]] 의 예측값 : 10.332012176513672






