import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import random_correlation
from tqdm import tqdm

from curves import *
from volatility import *
from correlation import *

class LiborMarketModel:
    def __init__(self, options):
        self.dt = 0.5
        self.df_interp = options['df_interp']
        self.r_interp = options['r_interp']
        self.vol_interp = options['vol_interp']
        self.corr_factors = options['corr_factors']
        self.T = options['T']
    def calibrate(self, spot_data):
        #Input: forward rates, cap volatilities and correlation matrix
        self.libor_curve = spot_data['libor_curve']
        self.ois_curve = spot_data['ois_curve']
        vol_boost =  CapletVolStrip(spot_data, self.vol_interp)
        t, stripped_vol = vol_boost.strip(spot_data['cap_vol'][0],spot_data['cap_vol'][1], display=False)
        self.vol_func = ParametricVolatility()
        self.vol_func.fit(t,stripped_vol)

        rho = spot_data['historical_rates'].iloc[:,0:int(self.T*2)]
        self.corr_func = CorrelationReduction(rho)
        self.rho = self.corr_func.factor_reduction(n_factors=self.corr_factors)
        self.dZ = self.corr_func.correlated_dZ
    def simulate(self, n_sim=100, seed = 1):
        #np.random.seed(seed)
        self.last_tenor = self.T
        tau = self.dt
        self.tenors = np.arange(0.5,self.last_tenor+0.5,0.5)
        self.time_steps = np.arange(self.dt,self.tenors[-1]+self.dt,self.dt)
        f0 = self.libor_curve.forward_rate(self.tenors-0.5,self.tenors)

        ois_f0 = self.ois_curve.forward_rate(self.tenors-0.5,self.tenors)
        lois = f0-ois_f0

        f = np.zeros(shape=(self.time_steps.shape[0]+1,self.tenors.shape[0],n_sim))
        P_OIS = np.zeros_like(f)

        dZ = self.dZ(steps=self.tenors.shape[0],n_sim=n_sim)
        for k in tqdm(range(n_sim)):
            f[0,:,k] = f0
            P_OIS[0,:,k] = np.cumprod(1/(1+tau*(f[0,:,k]-lois)))
            for i, t in enumerate(self.time_steps):
                live_T = self.tenors>t
                vol = self.vol_func(t,self.tenors[live_T])
                p = self.rho[-1,:][live_T]
                f_k = f[i,:,k][live_T]
                tmp = (tau*vol[1:]*f_k[1:]*p[1:])/(1+tau*f_k[1:])
                tmp = np.append(tmp,0) #terminal measure has 0 drift
                u_k_1 = np.flipud(np.flipud(tmp).cumsum())

                df = (u_k_1 - 0.5*vol**2)*self.dt+vol*np.sqrt(self.dt)*dZ[i,:,k][live_T]
                f_k_ = f_k*np.exp(df)
                #if a forward rate has matured, adapt array size
                #second drift
                tmp = (tau*vol[1:]*f_k_[1:]*p[1:])/(1+tau*f_k_[1:])
                tmp = np.append(tmp,0) #terminal measure has 0 drift
                u_k_2 = np.flipud(np.flipud(tmp).cumsum())
                u_k = (u_k_1+u_k_2)/2
                df = (u_k - 0.5*vol**2)*self.dt+vol*np.sqrt(self.dt)*dZ[i,:,k][live_T]
                f[i+1,:,k][live_T] = f_k*np.exp(df)
                P_OIS[i+1,:,k][live_T] = np.cumprod(1/(1+tau*(f_k*np.exp(df)-lois[live_T])))
        f[f==0] = np.nan
        P_OIS[P_OIS==0] = np.nan
        self.f = f
        self.P_OIS = P_OIS
    def plot_simulation(self, n_sim=0):
        fig = plt.figure(figsize=(10,10))
        X, Y = np.meshgrid(self.tenors,self.time_steps)

        plt.figure(0)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, self.f[1:,:,n_sim]*100, cmap='viridis',edgecolor='green')
        ax.set_xlabel('Tenor')
        ax.set_ylabel('Time Step')
        ax.set_zlabel('Rate')
        ax.view_init(40, 90)

        plt.figure(1)
        plt.title('Spot Curve')
        plt.xlabel('YF')
        plt.ylabel('r')

        plt.plot(np.arange(0,self.last_tenor,0.5),self.f[0,:,n_sim])
        plt.show()

def main():
    libor_curve, ois_curve = curve_builder()
    cap_vol = pd.read_excel('project_data.xlsx',sheet_name='caps vol', index_col='Tenor')
    t, vol = cap_vol.index.values, cap_vol.ATM.values/100
    historical_rates = pd.read_excel('project_data.xlsx',sheet_name='historical f', index_col='Date')

    spot_data = {'ois_curve': ois_curve,
                'libor_curve':libor_curve,
                'cap_vol': (t, vol),
                'historical_rates': historical_rates}
    options = {'r_interp':  {'kernel':'MCS','extrapolate':True},
               'df_interp': {'kernel':'MCS','extrapolate':True},
               'vol_interp':{'kernel':'MCS','extrapolate':True},
               'T': 4,
               'corr_factors': 'all'}

    model = LiborMarketModel(options)
    model.calibrate(spot_data)
    model.simulate()
    return model
def simulated_model(n_sim=100):
    model = main()
    model.simulate(n_sim)
    return model
def black_compare():
    np.random.seed(1)
    libor_curve, ois_curve = curve_builder()
    cap_vol = pd.read_excel('project_data.xlsx',sheet_name='caps vol', index_col='Tenor')
    t, vol = cap_vol.index.values, cap_vol.ATM.values/100
    historical_rates = pd.read_excel('project_data.xlsx',sheet_name='historical f', index_col='Date')
    spot_data = {'ois_curve': ois_curve,
                'libor_curve':libor_curve,
                'cap_vol': (t, vol),
                'historical_rates': historical_rates}
    options = {'r_interp':  {'kernel':'MCS','extrapolate':True},
               'df_interp': {'kernel':'MCS','extrapolate':True},
               'vol_interp':{'kernel':'MCS','extrapolate':True},
               'corr_factors': 'all'}
    vol_boost =  CapletVolStrip(spot_data, options['vol_interp'])
    t, stripped_vol = vol_boost.strip(spot_data['cap_vol'][0],spot_data['cap_vol'][1], display=False)
    vol_func = ParametricVolatility()
    vol_func.fit(t,stripped_vol)

    amount = 1000000
    lmm_price = []
    bs_price = []
    bs_vol = []
    lmm_vol = []
    for i in range(1,10,1):
        options['T'] = i # 2y
        model = LiborMarketModel(options)
        model.calibrate(spot_data)
        model.simulate(n_sim=100000)
        f = libor_curve.forward_rate(i-0.5,i)
        df = ois_curve.discount(i)
        tmp = model.f[-2,-1,:]-f
        tmp[tmp<0] = 0
        L = np.mean(tmp)
        lmm_price.append(L*0.5*amount*df)

        vol = vol_func.caplet_vol(0,i-0.5)
        bs_price.append(vol_boost.black_price(0,i-0.5,f,f,vol)*0.5*amount*df)

    lmm_price = np.array(lmm_price)
    bs_price = np.array(bs_price)
    print(pd.DataFrame({'LMM':lmm_price,'BS':bs_price,'%diff':(lmm_price/bs_price-1)*100}))


if __name__ == '__main__':
    black_compare()
