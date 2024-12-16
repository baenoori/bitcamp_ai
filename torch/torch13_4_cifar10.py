import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

path = './study/torch/_data'
train_dataset = CIFAR10(path, train=True, download=True)
test_dataset = CIFAR10(path, train=False, download=True)

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.size())    # torch.Size([50000, 32, 32, 3]) torch.Size([50000])
print(np.min(x_train.cpu().numpy()), np.max(x_train.cpu().numpy())) # 0.0 1.0

x_train, x_test = x_train.reshape(-1, 32*32*3), x_test.reshape(-1, 32*32*3)

train_dset =  TensorDataset(x_train, y_train)
test_dset =  TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loder = DataLoader(train_dset, batch_size=32, shuffle=False)

#2. 모델 
class DNN(nn.Module):       # class ()괄호 안에 들어가는건 상속 
    def __init__(self, num_features):
        super().__init__()
        # super(self,DNN).__init__()    # 위 아래 동일
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.output_layer = nn.Linear(32,10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = DNN(32*32*3).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4)   # 0.0001

def train(model, criterion, optimizer, loader):
    # model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)     # y = xw + b
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()

        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()     # y_predict == y_batch : True, False으로 결과 나옴

        epoch_loss += loss.item() 
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)

def evalutate(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)
# loss, acc = model.evaluate(x_test, y_test)

EPOCH = 100
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evalutate(model, criterion, test_loder)
    
    print(f'epoch : {epoch}, loss : {loss:.4f}, acc : {acc:.3f}, val_loss : {val_loss:.4f}, val_acc : {val_acc:.3f}')

#4. 평가, 예측
loss, acc = evalutate(model, criterion, test_loder)
print("================================================================================")
print('최종 Loss :', loss)
print('최종 acc :', acc)

# ========================================
# 최종 Loss : 1.4016029520516813
# 최종 acc : 0.49124280230326295