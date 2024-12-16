import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

print(aaa)

# for 문 돌려서 만들기 (함수 안에서 만들기)
import matplotlib.pyplot as plt

def outlier(data_out):    
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i],[25,50,75])
        # print('1 사분위 :', quartile_1)     # 4.0
        # print('q2 :', q2)                   # 7.0
        # print('3사분위 :', quartile_3)      # 10.0
        iqr = quartile_3 - quartile_1       # 10.0 - 4.0 = 6.0
        # print('IQR :', iqr)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        
        plt.subplot(2, 1, i+1)
        aaa = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))
        plt.boxplot(aaa)
        plt.axhline(iqr, color='red', label='IQR')

    plt.show()

outlier(aaa)
# outliers_loc , iqr = outlier(aaa)
# print('이상치의 위치 :', outliers_loc)
    
# subplot 형태로 컬럼 별로
# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.axhline(iqr, color='red', label='IQR')
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 생성
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50], 
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T
aaa = pd.DataFrame(aaa, columns=["Column1", "Column2"])

# 이상치 탐지 함수 정의
def outliers(data):
    num_cols = len(data.columns)
    fig, axes = plt.subplots(num_cols, 1, figsize=(8, 6 * num_cols))  # subplot 생성
    if num_cols == 1:  # 열이 1개인 경우, axes를 리스트로 변환
        axes = [axes]

    for i, col in enumerate(data.columns):
        print(f"=== {col} ===")
        col_data = data[col]
        quartile_1, q2, quartile_3 = np.percentile(col_data, [25, 50, 75])
        
        print("1사분위:", quartile_1)
        print("중앙값(q2):", q2)
        print("3사분위:", quartile_3)
        
        iqr = quartile_3 - quartile_1
        print("IQR:", iqr)
        
        low_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        
        # 이상치 인덱스 출력
        outliers = col_data[(col_data < low_bound) | (col_data > upper_bound)]
        print("이상치:")
        print(outliers)
        
        # 박스플롯 시각화
        ax = axes[i]
        ax.boxplot(col_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax.axvline(low_bound, color='red', linestyle='--', label="Lower Bound")
        ax.axvline(upper_bound, color='blue', linestyle='--', label="Upper Bound")
        ax.set_title(f"Boxplot of {col}")
        ax.legend()

    plt.tight_layout()
    plt.show()

# 이상치 탐지 실행
outliers(aaa)


