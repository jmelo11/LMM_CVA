import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ratemodels import *

class Swap6M:
    def __init__(self,years,side):
        self.y = years
        self.side_ = side
        self.coupon = 0
    def set_eval_date(self, t):
        self.t=t
        #Condiciones originales
        self.yf = np.arange(0.5,self.y+0.5,0.5)
        if t in self.yf:
            self.coupon = 1
        else:
            self.coupon = 0
        self.acc_yf = np.full(self.y*2,0.5)
        self.yf = self.yf - t
        time = (self.yf>0)
        self.yf = self.yf[time]
        paid = np.size(time)-np.count_nonzero(time)
        self.acc_yf = self.acc_yf[time]
        return 0
    @property
    def fixed_rate(self):
        return self._fixed_rate
    @property
    def years(self):
        return self.y
    @property
    def side(self):
        return self.side_
    @fixed_rate.setter
    def fixed_rate(self, arg):
        self._fixed_rate = arg
class IRS(Swap6M):
    def __init__(self, years, model,side=1):
        super().__init__(years, side)
        self.model = model
        self.par_rate()
    def par_rate(self):
        self.set_eval_date(0)
        fwd = self.model.f[0,:,0]
        df = self.model.P_OIS[0,:,0]
        self._fixed_rate = np.sum(self.acc_yf*fwd*df)/np.sum(self.acc_yf*df)
        return self._fixed_rate
    def simulate_MTM(self):
        f = self.model.f
        dfs = self.model.P_OIS
        dt = self.model.dt
        mtm = np.zeros(shape=(f.shape[0],1,f.shape[2]))
        for i in range(self.acc_yf.shape[0]):
            self.set_eval_date(dt*i)
            for j in range(f.shape[2]):
                df = dfs[i,:,j]
                df = df[df!=np.nan][i:i+self.acc_yf.shape[0]]
                fwd = f[i,:,j]
                fwd = fwd[fwd!=np.nan][i:i+self.acc_yf.shape[0]]
                if self.side_ == 0:
                    mtm[i,0,j] = self._fixed_rate*np.sum(self.acc_yf*df) - np.sum(self.acc_yf*fwd*df)
                else:
                    mtm[i,0,j] = -self._fixed_rate*np.sum(self.acc_yf*df) + np.sum(self.acc_yf*fwd*df)
        self.mtm = mtm
        self.set_eval_date(0)
        return mtm
    def CVA(self, cds, ois, recovery = 0.4):
        term_struct = np.arange(0,self.y+0.5,0.5)
        self.simulate_MTM()
        EE = self.mtm.copy()
        EE[EE<0] = 0
        NEE = self.mtm.copy()
        NEE[NEE>0] = 0

        self.EE = np.array([np.mean(EE[i,0,:]) for i in range(EE.shape[0])])
        self.NEE = np.array([np.mean(NEE[i,0,:]) for i in range(NEE.shape[0])])
        self.EPE = np.mean(self.EE)
        self.ENE = np.mean(self.NEE)

        dP = np.append(0,cds.dP(self.yf-0.5,self.yf))
        self.CVA = (1-recovery)*np.sum(self.EE*ois.discount(term_struct)*dP)
        return self.EE, self.NEE, self.EPE, self.ENE, self.CVA


def main():
    model = simulated_model()
    swap = IRS(years=5, model=model, side=0)
    mtm = swap.MTM()
    e_mtm = [np.mean(mtm[i,0,:]) for i in range(mtm.shape[0])]
    plt.plot(mtm[0,:,:])
    plt.show()
if __name__ == '__main__':
    main()
