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
# 045  아모레  3500  6000s
# 023  네이버   100  1500

print("======================================================")
print('시가가 1100원 이상이 행을 모두 출력')

cond1 = df['시가'] >= '1100'
print(cond1)
print(df[cond1])
print(df.loc[cond1])
# print(df.iloc[cond1]) # error

######## 1100원 이상만 뽑아서 새로 df2 만들기 
######## 이 방식을 제일 많이 쓸거임 
df2 = pd.DataFrame()
df2 = df[df['시가']>= '1100']
print(df2)
#      종목명    시가    종가
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000


##### 위에서 종가 컬럼만 뽑기 
print(df2['종가'])

df3 = df[df['시가']>='1100']['종가']
print(df3)

df4 = df.loc[df['시가']>='1100']['종가']
print(df4)

# df5 = df[df['시가']>='1100', '종가']
# print(df5)  # error

df6 = df.loc[df['시가']>='1100', '종가']
print(df6)

