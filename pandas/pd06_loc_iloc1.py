import pandas as pd
print(pd.__version__)   

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500'],
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

# print(df[0])  # error
# print(df['031'])  # error
print(df['시가'])   # 판다스 열행이기 때문에 컬럼이 기준 !!

### 아모레 출력 ###
# print(df[3, 0]) # key error
# print(df['045', '종목명'])    # key error
# print(df['종목명','045])      # key error
print(df['종목명']['045'])      # 결과 나옴 
print('----------------------------')

# loc : 인덱스 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출
        # int loc로 외우기 !!
        
print('==========아모레 뽑기============')
print(df.iloc[3])
print(df.loc['045'])

print('==========네이버 뽑기============')
print(df.iloc[4])
print(df.loc['023'])

print('==========아모레 종가 뽑기============')
print(df.iloc[3]['종가'])       # 6000
print(df.loc['045', '종가'])    # 6000

print(df.iloc[3][2])
print(df.iloc[3,2])
print(df.iloc[3].iloc[2])

print(df.loc['045'][2])
# print(df.loc['045', 2])   # error

print(df.iloc[3]['종가'])
# print(df.iloc[3, '종가'])   # error

print(df.loc['045'].iloc[2])
print(df.iloc[3].loc['종가'])
