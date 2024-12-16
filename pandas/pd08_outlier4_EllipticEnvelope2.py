import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T


# for 문 돌려서 열별로 찾게 수정 ! 


from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.3)           # 30%를 이상치로 간주
           #  defualt = 0.1

for i in range(aaa.shape[1]):
    # aaa[i] = aaa[i].reshape(-1,1)
    outliers = EllipticEnvelope()
    a = aaa[:, i].reshape(-1,1)
    outliers.fit(a)
    results = outliers.predict(a)
    print(results)      

