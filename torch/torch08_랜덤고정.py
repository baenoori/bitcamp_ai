import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

################### 랜덤 고정 ###################
SEED = 0
import random
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch cuda 시드 고정
####################################################

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)
# print(x.shape, y.shape)     # torch.Size([150, 4]) torch.Size([150])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=0,
                                                    stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # sparse cross entropy
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    # print('epoch : {}, loss : {:.8f}'.format(epoch, loss))
    print(f'epoch : {epoch}, loss : {loss:.8f}')

#4. 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print('loss :', loss)

############### acc 출력 ###############
y_predict = model(x_test)
acc = accuracy_score(y_test.cpu().numpy(), np.argmax(y_predict.detach().cpu().numpy(), axis=1))
print('acc_score :', acc)

# print(y_predict[:5])
# tensor([[-10.3642,  -4.3832,  10.1960],
#        [ -8.1778,  -3.3115,   7.9973],
#        [ 21.0935,  -3.3122, -40.7658],
#        [ 21.0603,  -2.4252, -42.0654],
#        [-16.4169,  24.2528, -13.0619]], device='cuda:0',
#       grad_fn=<SliceBackward0>)
y_predict = torch.argmax(model(x_test), dim=1)
# print(y_predict[:5])
# tensor([2, 2, 0, 0, 1], device='cuda:0')

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))
print(f'accuracy : {score:.4f}')

score2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuracy : {:.4f}'.format(score2))
print(f'accuracy : {score2:.4f}')


# loss : 0.4771515727043152
# acc_score : 0.9473684210526315
# accuracy : 0.9474
# accuracy : 0.9474
# accuracy : 0.9474
# accuracy : 0.9474

# loss : 1.517416603746824e-05  # seed 0
# acc_score : 1.0
# accuracy : 1.0000
# accuracy : 1.0000
# accuracy : 1.0000
# accuracy : 1.0000