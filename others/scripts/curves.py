from interpolation import *
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

class Curve(ABC):
    def discount(self,t):
        return 1/(1+self.zero_interp(t)*t)
    def forward_rate(self,t, T):
        df_T = self.discount(T)
        df_t = self.discount(t)
        return (np.log(df_t) - np.log(df_T))/(T-t)
    def print_curve(self, plot=True, freq=0.50):
        eval_date = ql.Settings.instance().evaluationDate+4
        t = np.arange(freq,20+freq,freq)
        dates = [eval_date+ql.Period(6*i,ql.Months) for i in range(1, t.shape[0]+1)]
        #df = self.discount(t)
        df = [self.ql_curve.discount(d) for d in dates]
        fwd = [self.ql_curve.forwardRate(d-ql.Period(6,ql.Months),d, ql.Actual360(), ql.Compounded, ql.Semiannual).rate() for d in dates]
        zero = self.forward_rate(0,t)
        #fwd = self.forward_rate(t-freq,t)
        print(pd.DataFrame({'Dates':dates,'YF':t,'DF':df,'Zero Rate':zero,'Forward Rate': fwd}))
        if plot==True:
            plt.subplot(1,3,1)
            plt.title('Discount Factors')
            plt.xlabel('Year Fractions')
            plt.ylabel('DF')
            plt.plot(t,df)

            plt.subplot(1,3,2)
            plt.title('Zero Rates')
            plt.xlabel('Year Fractions')
            plt.ylabel('r')
            plt.plot(t,zero)

            plt.subplot(1,3,3)
            plt.title('Forward Rates')
            plt.xlabel('Year Fractions')
            plt.ylabel('r')
            plt.plot(t,fwd)
            plt.show()
class OISCurve(Curve):
    def __init__(self, options):
        self.df_interp = options['df_interp']
        self.r_interp = options['r_interp']
    def boostrap(self, t, r, coupon_freq=1):
        '''
            USD OIS <= 1Y Maturity -> Period: tenor payment
            USD OIS > 1Y Maturity -> Anual payments
        '''
        par_rates = Interpolator(t, r, options=self.r_interp)
        self.T = int(t[-1])
        yfs = np.arange(coupon_freq,self.T+coupon_freq,coupon_freq)
        df = []
        for yf in yfs:
            if yf <= 1.5:
                df.append(1/(1+par_rates(yf)*yf))
            else:
                par_rate = par_rates(yf)
                df_T = (1-par_rate*np.sum(df)*coupon_freq)/(1+par_rate*coupon_freq)
                df.append(df_T)

        df = np.array(df)
        # add 1 and 180 discounts
        if coupon_freq==1:
            short_t = np.array([1/360,0.5])
            short_df = 1/(1+par_rates(short_t)*short_t)
            df =  np.concatenate((short_df, df), axis=0)
            yfs = np.concatenate((short_t, yfs), axis=0)
        zero_rates = (1/df-1)/yfs
        self.zero_interp = Interpolator(yfs, zero_rates, options=self.r_interp)
class ZeroRateCurve(Curve):
    def __init__(self, r_interp, options):
        self.options = options
        self.zero_interp = r_interp
class Libor3MCurve(Curve):
    '''
        Dual-Curve boostrapping, assumes both leg payment frequency is quarterly.
    '''
    def __init__(self, options):
        self.df_interp = options['df_interp']
        self.r_interp = options['r_interp']
    def boostrap(self, t, r, ois_curve):
        par_rates = Interpolator(t, r, options=self.r_interp)
        self.T = int(t[-1])
        yfs = np.arange(0.25,self.T+0.25,0.25)
        l = []
        df = 1
        for yf in yfs:
            if yf <= 1.5:
                x = 1/(1+par_rates(yf)*yf)
                l.append((df/x-1)/0.25)
                df = x
            else:
                yf = np.arange(0.25,yf+0.25,0.25)
                par_rate = par_rates(yf[-1])
                discounts = ois_curve.discount(yf)
                #print(yf[-1], discounts[:-1].shape, len(l))
                s = np.sum(l*discounts[:-1])
                f = (par_rate*np.sum(discounts)-s)/(discounts[-1])
                l.append(f)
        l = np.array(l)
        dfs = np.cumprod(1/(1+l*0.25))
        zero_rates = (1/dfs-1)/yfs
        self.zero_interp = Interpolator(yfs, zero_rates, options=self.r_interp)
class Libor6MCurve(Curve):
    def __init__(self, options):
        self.df_interp = options['df_interp']
        self.r_interp = options['r_interp']
    def boostrap(self, t, r, ois_curve):
        pass
class QLCurveHandle(Curve):
    def __init__(self, ql_curve):
        self.ql_curve = ql_curve
        self.f = np.vectorize(self.ql_curve.discount)
    def discount(self, t):
        if isinstance(t,(np.ndarray, np.generic)):
            return self.f(t)
        else:
            return self.ql_curve.discount(t)

class CDSCurve:
    def __init__(self, spot_data, options):
        self.df_interp = options['df_interp']
        self.r_interp = options['r_interp']
        self.ois_curve = spot_data['ois_curve']
    def boostrap(self, t, r, coupon_freq=0.5, recovery=0.4):
        #get survival probability
        par_rates = Interpolator(t, r, options=self.r_interp)
        self.T = int(t[-1])
        yfs = np.arange(coupon_freq,self.T+coupon_freq,coupon_freq)
        dps = np.array([1])
        L = (1-recovery)
        for yf in yfs:
            if yf <= 0.5:
                pi = par_rates(yf)
                dp = L/(L+pi*coupon_freq)
                dps = np.append(dps,dp)
            else:
                yf_ = np.arange(coupon_freq,yf+coupon_freq,coupon_freq)
                df = self.ois_curve.discount(yf_)
                pi = par_rates(yf)
                tmp = [df[i]*(L*dps[i-1]-(L+coupon_freq*pi)*dps[i]) for i in range(1,dps.shape[0])]
                dp = np.sum(tmp)
                dp /= (df[-1]*(L+coupon_freq*pi))
                dp += dps[-1]*L/(L+coupon_freq*pi)
                #dp = (df_1*(L-(L-coupon_freq*pi)*dps[-1])/(df_2*(L+coupon_freq*pi))) + dps[-1]*L/(L+coupon_freq*pi)
                dps = np.append(dps,dp)
        yfs = np.arange(0,self.T+coupon_freq,coupon_freq)
        self.df_interp = Interpolator(yfs, dps, options=self.df_interp)
        return yfs, dps
    def dP(self, t, T):
        return self.P(0,t) - self.P(0, T)
    def P(self, t, T):
        P_T = self.df_interp(T)
        P_t = self.df_interp(t)
        return P_T/P_t
    def hazard_rate(self, t, T):
        P_T = self.P(0,t)
        P_t = self.P(0,T)
        return (np.log(P_t) - np.log(P_T))/(T-t)
#OIS OK
def qlOIS():
    eval_date = ql.Date(23,12,2019)
    ql.Settings.instance().evaluationDate = eval_date
    helpers = []
    calendar = ql.UnitedStates()
    df = pd.read_excel(r'C:\Users\jmelo\OneDrive\Escritorio\Pega\cqf\final_project\project_data.xlsx',sheet_name='ois')
    index = ql.OvernightIndex('US_OIS',2,ql.USDCurrency(),calendar,ql.Actual360())
    for i in range(df.shape[0]):
        if df.at[i, 'Daycount'] =='ACT/360': convention = ql.Actual360()
        elif df.at[i, 'Daycount'] =='30I/360': convention = ql.Thirty360()
        term = int(df.at[i,'Term'])
        rate = ql.QuoteHandle(ql.SimpleQuote(float(df.at[i, 'Par']/100)))
        if df.at[i,'Unit'] == 'DY' or 'ACTDATE':
            period = ql.Period(term,ql.Days)
        if df.at[i,'Unit'] == 'WK':
            period = ql.Period(term, ql.Weeks)
        if df.at[i,'Unit'] == 'MO':
            period = ql.Period(term, ql.Months)
        if df.at[i,'Unit'] == 'YR':
            period = ql.Period(term, ql.Years)
        if df.at[i,'Freq'] < 1:
            helper = ql.DepositRateHelper(rate, period, 2, calendar, ql.Unadjusted, False, convention)
        else:
            helper = ql.OISRateHelper(2, period, rate, index)

        helpers.append(helper)
    curve = ql.PiecewiseCubicZero(eval_date, helpers, ql.Actual360())
    curve.enableExtrapolation()
    return curve
def qlLibor3mSwap(discount_curve):
        eval_date = ql.Date(23,12,2019)
        ql.Settings.instance().evaluationDate = eval_date
        helpers = []
        calendar = ql.UnitedStates()

        df = pd.read_excel(r'C:\Users\jmelo\OneDrive\Escritorio\Pega\cqf\final_project\project_data.xlsx',sheet_name='3m libor')
        index = ql.USDLibor(ql.Period(3, ql.Months))
        handle = ql.YieldTermStructureHandle(discount_curve)

        for i in range(df.shape[0]):
            if df.at[i, 'Daycount'] =='ACT/360': convention = ql.Actual360()
            elif df.at[i, 'Daycount'] =='30I/360': convention = ql.Thirty360()
            term = int(df.at[i,'Term'])
            rate = ql.QuoteHandle(ql.SimpleQuote(float(df.at[i, 'Par']/100)))
            if df.at[i,'Unit'] == 'DY' or 'ACTDATE':
                period = ql.Period(int(df.at[i,'Term']),ql.Days)
            if df.at[i,'Unit'] == 'WK':
                period = ql.Period(term, ql.Weeks)
            if df.at[i,'Unit'] == 'MO':
                period = ql.Period(term, ql.Months)
            if df.at[i,'Unit'] == 'YR':
                period = ql.Period(term, ql.Years)
            if df.at[i,'Freq'] < 1:
                helper = ql.DepositRateHelper(rate, period, 2, calendar, ql.Unadjusted, False, convention)
            else:
                helper = ql.SwapRateHelper(rate, period, calendar, ql.Semiannual, ql.Unadjusted, convention, index, ql.QuoteHandle(), ql.Period(0,ql.Days), handle)
            helpers.append(helper)

        curve = ql.PiecewiseCubicZero(eval_date, helpers, ql.Actual360())
        curve.enableExtrapolation()
        return curve
def qlLibor6mSwap(discount_curve):
        eval_date = ql.Date(23,12,2019)
        ql.Settings.instance().evaluationDate = eval_date
        helpers = []
        calendar = ql.UnitedStates()

        df = pd.read_excel(r'C:\Users\jmelo\OneDrive\Escritorio\Pega\cqf\final_project\project_data.xlsx',sheet_name='6m libor')
        index = ql.USDLibor(ql.Period(6, ql.Months))
        handle = ql.YieldTermStructureHandle(discount_curve)

        for i in range(df.shape[0]):
            if df.at[i, 'Daycount'] =='ACT/360': convention = ql.Actual360()
            elif df.at[i, 'Daycount'] =='30I/360': convention = ql.Thirty360()
            term = int(df.at[i,'Term'])
            rate = ql.QuoteHandle(ql.SimpleQuote(float(df.at[i, 'Par']/100)))
            if df.at[i,'Unit'] == 'DY' or 'ACTDATE':
                period = ql.Period(int(df.at[i,'Term']),ql.Days)
            if df.at[i,'Unit'] == 'WK':
                period = ql.Period(term, ql.Weeks)
            if df.at[i,'Unit'] == 'MO':
                period = ql.Period(term, ql.Months)
            if df.at[i,'Unit'] == 'YR':
                period = ql.Period(term, ql.Years)
            if df.at[i,'Freq'] < 1:
                helper = ql.DepositRateHelper(rate, period, 2, calendar, ql.Unadjusted, False, convention)
            else:
                helper = ql.SwapRateHelper(rate, period, calendar, ql.Semiannual, ql.Unadjusted, convention, index, ql.QuoteHandle(), ql.Period(0,ql.Days), handle)
            helpers.append(helper)

        curve = ql.PiecewiseCubicZero(eval_date, helpers, ql.Actual360())
        curve.enableExtrapolation()
        return curve
def curve_builder():
    ois_curve = qlOIS()
    libor_curve = QLCurveHandle(qlLibor6mSwap(discount_curve= ois_curve))
    ois_curve = QLCurveHandle(ois_curve)
    return libor_curve, ois_curve
def main():
    libor_curve, ois_curve = curve_builder()
    libor_curve.print_curve()
def test_cds():
    df = pd.read_excel(r'C:\Users\jmelo\OneDrive\Escritorio\Pega\cqf\final_project\project_data.xlsx',sheet_name='jpm cds', index_col = 'Tenor')
    options = {'r_interp':  {'kernel':'CS','extrapolate':True},
               'df_interp': {'kernel':'CS','extrapolate':True}}

    libor, ois = curve_builder()
    cds =  CDSCurve({'ois_curve':ois},options)
    t, dp = cds.boostrap(df.index,df.Spread/10000)
    print(t.shape, dp.shape)

if __name__ == '__main__':
    test_cds()
