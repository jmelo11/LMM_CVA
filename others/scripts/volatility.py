import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
from lib.curves import *

class ParametricVolatility:
    def __init__(self):
        super()
    def fit(self,T,vol, solver='Nelder-Mead',display=False):
        f = lambda x: self.caplet_vol(vol,T,x)
        x0 = np.array([0.5,0.5,0.5,0.5])
        res = minimize(f,x0, method=solver,options={'xtol': 1e-8, 'disp': display,'maxiter':100000})
        self.params = res.x
        return self.params
    def __call__(self,t,T):
        return self.instant_vol(t,T,self.params)
    @abstractmethod
    def instant_vol(self,t,T,params):
        return (params[0] + params[1]*(T-t))*np.exp(-params[2]*(T-t))+params[3]
    @abstractmethod
    def caplet_vol(self,vol,T,params):
        vol_f = np.array([0.5*self.instant_vol(0,i,params)**2 for i in T])
        vol_f = np.sqrt(np.cumsum(vol_f)/T)
        vol_mkt = np.array([0.5*v**2 for v in vol])
        vol_mkt = np.sqrt(np.cumsum(vol_mkt)/T)
        return np.sum((vol_mkt-vol_f)**2)
class CapletVolStrip:
    def __init__(self, spot_data, options):
        self.ois_curve = spot_data['ois_curve']
        self.libor_curve = spot_data['libor_curve']
        self.vol_interp = options
    #@abstractmethod
    def swap_rate(self,t,T,coupon_freq=1):
        proy_curve = self.libor_curve
        disc_curve = self.ois_curve

        t = np.arange(t,T+coupon_freq,coupon_freq)
        f = proy_curve.forward_rate(t-coupon_freq,t)
        df = disc_curve.discount(t)
        swap_rate = (df[0]-df[-1])/(coupon_freq*np.sum(df[1:]))
        return swap_rate
    @abstractmethod
    def black_price(self,t,T,f,k,vol_t,op_type='caplet'):
        #f: forward rate up to time T
        if op_type == 'caplet':
            w = 1
        elif op_type=='floorlet':
            w = -1
        d1 = (np.log(f/k)+(0.5*(vol_t)**2)*(T-t))/(vol_t*np.sqrt(T-t))
        d2 = (np.log(f/k)-(0.5*(vol_t)**2)*(T-t))/(vol_t*np.sqrt(T-t))
        return f*w*norm.cdf(w*d1)-k*w*norm.cdf(w*d2)
    def strip(self, t, vol, display=False):
        cap_vol = Interpolator(t,vol, self.vol_interp)
        proy_curve = self.libor_curve
        disc_curve = self.ois_curve
        self.T = int(t[-1])
        eval_t = np.arange(1.5,self.T+0.5,0.5)
        caplet_vol = [cap_vol(1)]
        for t in eval_t:
            term_struc = np.arange(0.5,t+0.5,0.5)
            vol = cap_vol(t)
            s = self.swap_rate(0.5,t, coupon_freq=0.5)
            fwd = proy_curve.forward_rate(term_struc-0.5,term_struc)
            cap = 0
            for f,c in zip(fwd[1:],term_struc[1:]):
                df = disc_curve.discount(c)
                bl = self.black_price(0.5,c,f,s,vol)
                cap += df*bl*0.5

            i = 0
            caplets = 0

            for f,c in zip(fwd[1:-1],term_struc[1:-1]):
                df = disc_curve.discount(c)
                bl = self.black_price(0.5,c,f,s,caplet_vol[i])
                i += 1
                caplets += df*bl*0.5

            obj = lambda v: (cap - caplets - disc_curve.discount(t)*self.black_price(0.5,t,fwd[-1],s,v)*0.5)**2
            res = minimize(obj, cap_vol(c), method='Nelder-Mead',options={'xtol': 1e-8, 'disp': display,'maxiter':10000})
            caplet_vol.append(res.x[0])
        self.caplet_vol = np.array(caplet_vol)
        self.t = np.arange(0.5,self.T,0.5)
        return self.t, self.caplet_vol
    def print_curve(self):
        print(pd.DataFrame({'YF':self.t,'CapletVol':self.caplet_vol}))
        plt.figure()
        plt.title('Caplet volatility term structure.')
        plt.xlabel('YF')
        plt.ylabel('Caplet Vol')
        plt.plot(self.t,self.caplet_vol)
        plt.show()
def main():
    libor_curve, ois_curve = curve_builder()
    spot_data = {'ois_curve': ois_curve,
                 'libor_curve':libor_curve}

    options = {'kernel':'CS','extrapolate':True}
    vol_data = pd.read_excel(r'C:\Users\jmelo\OneDrive\Escritorio\Pega\cqf\final_project\project_data.xlsx',sheet_name='caps vol', index_col='Tenor')
    vol_data = vol_data[0:11]
    vol_boost =  CapletVolStrip(spot_data, options)
    t, vol = vol_boost.strip(vol_data.index.values,vol_data['ATM'].values/100, display=False)

    vol_interp = Interpolator(vol_data.index.values-0.5,vol_data['ATM'].values/100, options)

    vol_f = ParametricVolatility()
    vol_f.fit(t,vol,display=True)
    cap_vol = vol_interp(t)
    parametric_vol = vol_f(0,t)
    print(pd.DataFrame({'T':t,'Cap Vol':cap_vol,'Caplet Vol':vol,'Parametric':parametric_vol}))

    plt.figure()
    plt.plot(t, cap_vol, label='Cap Vol')
    plt.plot(t, parametric_vol, label='Parametric Vol')
    plt.plot(t, vol, label='Caplet Vol')
    plt.legend(loc='lower right')
    plt.show()
    print(vol_f(1,1.5))
if __name__ == '__main__':
    main()
