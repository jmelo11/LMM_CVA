from interpolation import *
from abc import ABC
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

class Curve(ABC):
    def forward_rate(self,t, T):
        df_T = self.discount(T)
        df_t = self.discount(t)
        return (np.log(df_t) - np.log(df_T))/(T-t)
    def print_curve(self, plot=True, freq=0.50):
        eval_date = ql.Settings.instance().evaluationDate+4
        t = np.arange(freq,20+freq,freq)
        dates = [eval_date+ql.Period(6*i,ql.Months) for i in range(1, t.shape[0]+1)]
        #df = self.discount(t)
        df = np.array([self.ql_curve.discount(d) for d in dates])
        fwd = np.array([self.ql_curve.forwardRate(d-ql.Period(6,ql.Months),d, ql.Actual360(), ql.Compounded, ql.Annual).rate() for d in dates])
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
            plt.plot(t,zero*100)

            plt.subplot(1,3,3)
            plt.title('Forward Rates')
            plt.xlabel('Year Fractions')
            plt.ylabel('r')
            plt.plot(t,fwd*100)
            plt.show()
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
    def __init__(self, ois_curve, options):
        self.df_interp = options['df_interp']
        self.r_interp = options['r_interp']
        self.ois_curve = ois_curve
    def bootstrap(self, t, r, coupon_freq=0.5, recovery=0.4):
        #get survival probability
        par_rates = Interpolator(t, r, options=self.r_interp)
        self.T = int(t[-1])
        yfs = np.arange(coupon_freq,self.T+coupon_freq,coupon_freq)
        ps = np.array([1])
        L = (1-recovery)
        for yf in yfs:
            if yf <= 0.5:
                pi = par_rates(yf)
                p = L/(L+pi*coupon_freq)
            else:
                yf_ = np.arange(coupon_freq,yf+coupon_freq,coupon_freq)
                df = self.ois_curve.discount(yf_)
                pi = par_rates(yf)
                tmp = [df[i]*(L*ps[i-1]-(L+coupon_freq*pi)*ps[i]) for i in range(1,ps.shape[0])]
                p = np.sum(tmp)/(df[-1]*(L+coupon_freq*pi))+ps[-1]*L/(L+coupon_freq*pi)
            ps = np.append(ps,p)
        yfs = np.arange(0,self.T+coupon_freq,coupon_freq)
        self.df_f = Interpolator(yfs, ps, options=self.df_interp)
        return yfs, ps
    def dP(self, t, T):
        return self.P(0,t) - self.P(0, T)
    def P(self, t, T):
        P_T = self.df_f(T)
        P_t = self.df_f(t)
        return P_T/P_t
    def hazard_rate(self, t, T):
        P_T = self.P(0,t)
        P_t = self.P(0,T)
        return -(np.log(P_t) - np.log(P_T))/(T-t)
#OIS OK
def qlOIS():
    eval_date = ql.Date(23,12,2019)
    ql.Settings.instance().evaluationDate = eval_date
    helpers = []
    calendar = ql.UnitedStates()
    df = pd.read_excel('project_data.xlsx',sheet_name='ois')
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

        df = pd.read_excel('project_data.xlsx',sheet_name='3m libor')
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

        df = pd.read_excel('project_data.xlsx',sheet_name='6m libor')
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
def rates():
    libor_curve, ois_curve = curve_builder()
    libor_curve.print_curve()
def cds():
    df = pd.read_excel('project_data.xlsx',sheet_name='jpm cds', index_col = 'Tenor')
    options = {'r_interp':  {'kernel':'CS','extrapolate':True},
               'df_interp': {'kernel':'CS','extrapolate':True}}

    libor, ois = curve_builder()
    cds =  CDSCurve(ois, options)
    t, p = cds.bootstrap(df.index,df.Spread/10000)
    dp = cds.dP(0,t)
    hr = cds.hazard_rate(0,t)
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('$P$')
    plt.xlabel('Years')
    plt.ylabel('$P$(%)')
    plt.plot(t,p*100)

    plt.subplot(1,3,2)
    plt.title('$dP$')
    plt.xlabel('Years')
    plt.ylabel('$dP$(%)')
    plt.plot(t,dp*100)

    plt.subplot(1,3,3)
    plt.title('Hazard Rate ($\lambda$)')
    plt.xlabel('Years')
    plt.ylabel('$\lambda$(%)')
    plt.plot(t,hr*100)
    plt.show()
if __name__ == '__main__':
    cds()
