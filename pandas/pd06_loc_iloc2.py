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

print("============== iloc 아모레와 네이버의 시가 ===============")
print(df.iloc[3:5, 1])      # 행, 열
print(df.iloc[3:, 1])
# 045    3500
# 023     100

print(df.iloc[[2, 4], 1])   # LG와 네이버의 시가    # 특정 행만 뽑기 
# 033    2000
# 023     100
# print(df.iloc[3:, '시가']) # error
# print(df.iloc[[2, 4], '시가']) # error

print("============== loc 아모레와 네이버의 시가 ===============")
print(df.loc['045':'023', '시가'])      # 행, 열
print(df.loc['045': , '시가'])
# 045    3500
# 023     100

print(df.loc[['033', '023'], '시가'])   # LG와 네이버의 시가    # 특정 행만 뽑기 
# 033    2000
# 023     100
# print(df.loc['045':, 1]) # error
# print(df.loc[['033', '023'], 1]) # error

print("==================================================")
# print(df.loc['045', '033'][2])    # error
print(df.loc['045'].iloc[2])
print(df.loc['033':'023'].iloc[2])

# print(df.iloc[[2,4]][2])    # errorz
print(df.iloc[2].loc['시가'])    # 2000
print(df.iloc[2:4].iloc[2:4])



