import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)
# torch : 2.4.1+cu124 사용 device : cuda

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])
x_pre = np.array([12,13,14])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_pre = torch.FloatTensor(x_pre).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape) # torch.Size([7, 1]) torch.Size([7, 1])

#2. 모델 구성 
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1)
).to(DEVICE)

#3. 컴파일, 룬련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))

print("=========================================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pre = model(x)
        loss2 = criterion(y, y_pre)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', loss2)

results = model(x_pre)

print("[12,13,14] 의 예측값 :", results.detach().cpu().numpy())

# 최종 loss : 2.8421709430404007e-13
# [12,13,14] 의 예측값 : 
# [[12.000001]
#  [13.      ]
#  [14.000001]]


