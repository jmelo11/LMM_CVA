import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
from curves import *

class ParametricVolatility:
    def __init__(self):
        super()
    def fit(self,T,vol, solver='L-BFGS-B',display=False):
        self.T = T
        self.vol = vol
        f = lambda x: self.opt(vol,T+0.5,x)
        x0 = np.array([0.1,0.1,0.1,0.1])
        res = minimize(f,x0, method=solver,options={'disp': display,'maxiter':100000}, bounds=[(0,None),(0,None),(0,None),(0,None)])
        self.params = res.x
        return self.params
    def __call__(self,t,T):
        return self.instant_vol(t,T,self.params)
    @abstractmethod
    def instant_vol(self,t,T,params):
        return (params[0] + params[1]*(T-t))*np.exp(-params[2]*(T-t))+params[3]
    def exact_instant_vol(self,t,T,params):
        vol_interp = Interpolator(self.T,self.vol, {'kernel':'linear','extrapolate':True})
        s = vol_interp(T-t)/self.caplet_vol( t, T, params)
        return ((params[0] + params[1]*(T-t))*np.exp(-params[2]*(T-t))+params[3])*s
    def caplet_vol(self, t, T, params):
        a, b, c ,d = params[0],params[1],params[2],params[3]
        a_sqr = a**2
        b_sqr = b**2
        c_sqr = c**2
        d_sqr = d**2
        tau = T-t
        exp_term = np.exp(-c*tau)
        exp_plus_term = np.exp(c*tau)
        term1 = -2*c_sqr*(a_sqr+4*a*d*exp_plus_term-2*c*d_sqr*tau*exp_plus_term*exp_plus_term)
        term2 = 2*b*c*(2*a*c*tau+a+4*d*exp_plus_term*(c*tau+1))
        term3 = b_sqr*(-(2*c_sqr*tau*tau+2*c*tau+1))
        v_T = 1/(4*c**3)*exp_term*exp_term*(term1-term2+term3)

        tau = 0
        exp_term = np.exp(-c*tau)
        exp_plus_term = np.exp(c*tau)
        term1 = -2*c_sqr*(a_sqr+4*a*d*exp_plus_term-2*c*d_sqr*tau*exp_plus_term*exp_plus_term)
        term2 = 2*b*c*(2*a*c*tau+a+4*d*exp_plus_term*(c*tau+1))
        term3 = b_sqr*(-(2*c_sqr*tau*tau+2*c*tau+1))
        v_0 = 1/(4*c**3)*exp_term*exp_term*(term1-term2+term3)
        return np.sqrt((v_T-v_0)/(T-t))
    def opt(self,vol,T,params):
        return np.sum((vol-self.caplet_vol(0,T,params))**2)
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
    vol_boost =  CapletVolStrip(spot_data, options)
    t, vol = vol_boost.strip(vol_data.index.values,vol_data['ATM'].values/100, display=False)

    vol_interp = Interpolator(vol_data.index.values-0.5,vol_data['ATM'].values/100, options)

    vol_f = ParametricVolatility()
    vol_f.fit(t,vol,display=True)
    cap_vol = vol_interp(t)
    parametric_vol = vol_f.caplet_vol(0,t,vol_f.params)
    instant_vol = vol_f(0,t)

    print(pd.DataFrame({'T':t,'Cap Vol':cap_vol,'Caplet Vol':vol,'Parametric':parametric_vol,'Instant': instant_vol}))

    plt.figure()
    plt.plot(t, cap_vol, label='Cap Vol')
    plt.plot(t, parametric_vol, label='Parametric Capelt Vol')
    plt.plot(t, vol, label='Real Caplet Vol')
    plt.plot(t, instant_vol, label='Instantaneous Vol')
    plt.legend(loc='upper right')
    plt.show()
    print(vol_f.params)

if __name__ == '__main__':
    main()
