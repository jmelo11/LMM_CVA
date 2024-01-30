import numpy as np
import pandas as pd
from lib.correlation import *

class CorrelationReduction:
    def __init__(self, df):
        self.data = df
    def corr(self, return_as='DataFrame'):
        self.r = np.log(self.data).diff()
        if return_as =='DataFrame':
            return self.r.corr()
        else:
            return self.r.corr().values
    def rank_reduction(self,n_factors=3):
        rho = self.corr(return_as='array')
        v, A = np.linalg.eig(rho)

        if n_factors=='all':
            self.n_factors = rho.shape[0]
            B = A @ np.diag(np.sqrt(v))
        else:
            self.n_factors = n_factors
            v[n_factors:] = 0
            B = A @ np.diag(np.sqrt(v))
            B = B[:,0:n_factors]

        reduced = B @ B.T
        for i in range(reduced.shape[0]):
            for j in range(reduced.shape[1]):
                reduced[i,j] = reduced[i,j]/(np.sqrt(reduced[i,i]*reduced[j,j]))
        self.reduced = reduced
        self.B = B
        return reduced
    def correlated_dZ(self,steps=10,n_sim=100):
        phi = np.zeros(shape=(steps,self.B.shape[0],n_sim))
        for i in range(n_sim):
            W = np.random.normal(0,1,size=(steps,self.n_factors))
            tmp = (self.B @ W.T).T
            phi[:,:,i] = tmp
        return phi
def main():
    df = pd.read_excel(r'C:\Users\jmelo\OneDrive\Escritorio\Pega\cqf\final_project\project_data.xlsx',sheet_name='historical f', index_col='Date')
    correlation = CorrelationReduction(df)
    correlation.rank_reduction(n_factors=3)
    dZ = correlation.correlated_dZ(n_sim=1)
    print(pd.DataFrame(dZ[:,:,0]))


if __name__ == '__main__':
    main()
