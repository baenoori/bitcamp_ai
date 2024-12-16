import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)


# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)
# print(x.shape, y.shape)     # torch.Size([150, 4]) torch.Size([150])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=5123,
                                                    stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, x_test.shape)  # torch.Size([1347, 64]) torch.Size([450, 64])  
print(y_train.shape, y_test.shape)  # torch.Size([1347]) torch.Size([450])
print(y_train.unique()) # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device='cuda:0')

#2. 모델
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
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

y_predict = torch.argmax(model(x_test), dim=1)
score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))
print(f'accuracy : {score:.4f}')

score2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuracy : {:.4f}'.format(score2))
print(f'accuracy : {score2:.4f}')

# loss : 0.5445559024810791
# acc_score : 0.9666666666666667
# accuracy : 0.9667
# accuracy : 0.9667
# accuracy : 0.9667
# accuracy : 0.9667
