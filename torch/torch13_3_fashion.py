import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import FashionMNIST

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

path = "./study/torch/_data/"
train_dataset = FashionMNIST(path, train=True, download=True)
test_dataset = FashionMNIST(path, train=False, download=True)

print(train_dataset)
print(type(train_dataset))  # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[0])     # (<PIL.Image.Image image mode=L size=28x28 at 0x14B0B2E8500>, 5)

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

print(x_train)
print(y_train)
print(x_train.shape, y_train.size())        # torch.Size([60000, 28, 28]) torch.Size([60000])

print(np.min(x_train.numpy()), np.max(x_train.numpy())) # 0.0 1.0

x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)
# torch에서의 reshape 는 view ! view는 연속적인 데이터에서만 사용 (ex. 1 2 3 4 5 6), view가 성능이 좀 더 좋음
print(x_train.shape, x_test.size())     # torch.Size([60000, 784]) torch.Size([10000, 784])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

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
    
model = DNN(784).to(DEVICE)

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
# 최종 Loss : 0.3493358797550201
# 최종 acc : 0.8749333333333333