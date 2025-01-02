# [실습] 얼리스타핑 적용하려면 어떻게 해야할까?

# 1. 최소 값을 넣을 변수 하나, 카운트할 변수 하나 준비
# 2. 다음 에포에 값과 최소 값을 비교, 최소 값이 갱신되면
#    그 다음 변수에 최소 값 넣고, 카운트 변수 초기화
# 3. 갱신이 안되면 카운트 변수 +=1
#    카운트 변수가 내가 원하는 얼리 스타핑에 도달하면 for문 stop

x = 10
y = 10
w = 0.001
# lr = 0.01 # 핑퐁
lr = 0.001 #갱신 잘 됨
# lr = 0.0001 # 갱신 안 됨
epochs = 10000

min =  float('inf')
count = 0

for i in range(epochs):
    
    hypothesis = x * w
    loss = (hypothesis - y ) **2 # bias 고려하지 않은 mse
    
    print( 'Loss : ', round(loss, 4), '\tPredict : ', round(hypothesis, 4))
    
    up_predict = x*(w + lr)
    up_loss = (y - up_predict) **2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) **2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr
    
    if(loss < min): 
        min = loss
          
    else :
        count +=1
        
    if(count == 10):
        print('Loss : ', round(loss, 4), '\t Predict : ', round(hypothesis, 4))
        break
    