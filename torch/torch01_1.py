import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)
# print(x.shape)      # torch.Size([3])
# print(x.size())     # torch.Size([3]) torch에서는 size를 많이 씀

x = torch.FloatTensor(x).unsqueeze(1)   # (3,) -> (3,1) , tensor로 바꿔주고 shape 바꿔줌, 행렬형태로 
# print(x)
# tensor([[1.], 
#         [2.], 
#         [3.]])
y = torch.FloatTensor(y).unsqueeze(1)   # (3,) -> (3,1)
print(x.shape, y.shape)     # torch.Size([3, 1]) torch.Size([3, 1])
print(x.size(), y.size())   # torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델 구성 
# model = Sequential()
# model.add(Dense(1, input_dim=1))   # (output, input)
model = nn.Linear(1, 1)     # (input, output)       # y = xw + b

#3. 컴파일, 훈련
# model.compile(loss="mse", optimizer='adam')
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)      # parameters 명시 필요 

def train(model, criterion, optimizer, x, y):
    # model.train()       # 가중치 갱신이 들어가는 훈련모드 상태 (디폴트)
    optimizer.zero_grad()   # 각 배치마다 기울기를 초기화(0으로)하여, 기울기 누적에 의한 문제 해결
    
    hypothesis = model(x)         # 가설을 잡음, y'로 생각 (y = xw + b)
    
    loss = criterion(hypothesis, y)     # loss = mse() = (y - hypothesis)^2/n
    
    loss.backward()     # 기울기(gradient)값 계산까지   # 순전파 진행 후 역전파 시작점, 기울기 : loss를 weight값으로 미분한 값
    optimizer.step()    # 가중치(w) 갱신               # 역전파 끝    
    
    return loss.item()     # tenor 값으로 빼줌 -> 우리가 볼 수 있는 값으로 출력하기 위해 loss.item()을 해줌

epochs = 20000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))        # verbose 

print("=====================================================")

# 4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y):       # 가중치 갱신이 필요없어서 optimizer X 
    model.eval()       # 평가모드, 가중치와 기울기 갱신을 하지 말아라 라는 의미
    
    with torch.no_grad():                       # graident가 갱신될 수 있어서 설정
        y_predict = model(x)
        loss2 = criterion(y, y_predict)         # loss의 최종값
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print("최종 loss :", loss2)        

result = model(torch.Tensor([[4]]))       # model의 가중치가 최상의 가중치가 저장되어 있어 tensor형태의 값으로 넣어줌
print("4의 예측값 :", result)              # 4의 예측값 : tensor([[3.9947]], grad_fn=<AddmmBackward0>)      # item 미사용시 gradient도 나옴 
print("4의 예측값 :", result.item())       # 4의 예측값 : 3.9947216510772705


# epoch: 1999, loss: 6.993600891291862e-06
# epoch: 2000, loss: 6.959507572901202e-06
# =====================================================
# 최종 loss : 6.92642379362951e-06
# 4의 예측값 : tensor([[3.9947]], grad_fn=<AddmmBackward0>)
# 4의 예측값 : 3.9947216510772705