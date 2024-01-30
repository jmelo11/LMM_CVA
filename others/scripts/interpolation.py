import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd

class Interpolator:
    def __init__(self,x,y, options):
        self.kernel = options['kernel']
        self.ex = options['extrapolate']
        self.x = x
        self.y = y
    def __call__(self, x_hat):
        if self.kernel == 'linear':
            interpolator = self.linear_interpolation
        elif self.kernel == 'MCS':
            interpolator = self.MCS_interpolation
        elif self.kernel == 'CS':
            interpolator = CubicSpline(self.x,self.y)
        else:
            raise ValueError('Unsupported kernel.')

        if isinstance(x_hat,(np.ndarray, np.generic)):
            if x_hat.size>0:
                f = np.vectorize(interpolator)
                return f(x_hat)
            else:
                return interpolator(x_hat)
        else:
            return interpolator(x_hat)
    @abstractmethod
    def linear_interpolation(self, x_hat):
        if x_hat>self.x[-1]:
            if self.ex ==True:
                return self.y[-2] + (x_hat-self.x[-2])/(self.x[-1]-self.x[-2])*(self.y[-1]-self.y[-2])
            else:
                raise ValueError('Extrapolation not enabled.')
        else:
            pos = np.searchsorted(self.x,x_hat)-1
            a = ((x_hat - self.x[pos])/(self.x[pos+1]-self.x[pos]))*self.y[pos+1]
            b = ((self.x[pos+1] - x_hat)/(self.x[pos+1]-self.x[pos]))*self.y[pos]
            return a+b

    @abstractmethod
    def MCS_interpolation(self, x_hat):
        i = np.searchsorted(self.x,x_hat)-1
        if i<0:
            return self.y[0]
        if x_hat>=self.x[-1]:
            if self.ex ==True:
                return self.y[-2] + (x_hat-self.x[-2])/(self.x[-1]-self.x[-2])*(self.y[-1]-self.y[-2])
            else:
                raise ValueError('Extrapolation not enabled.')
        h = self.x[i+1]-self.x[i]
        h_ = self.x[i]-self.x[i-1]
        s = (self.y[i+1]-self.y[i])/h
        s_= (self.y[i]-self.y[i-1])/h_
        a = (self.y_(i)+self.y_(i+1)-2*s)/h**2
        b = (3*s-self.y_(i)*2-self.y_(i+1))/h
        c = self.y_(i)
        d = self.y[i]
        f = a*(x_hat-self.x[i])**3+b*(x_hat-self.x[i])**2+c*(x_hat-self.x[i])+d
        return f
    @abstractmethod
    def y_(self, i):
        h = self.x[i+1]-self.x[i]
        h_ = self.x[i]-self.x[i-1]
        s = (self.y[i+1]-self.y[i])/h
        s_= (self.y[i]-self.y[i-1])/h_
        p = (s_*h+s*h_)/(h_+h)
        if s<=0 or s_<=0:
            y_ = 0
        elif abs(p)>2*abs(s_) or abs(p)>2*abs(s):
            y_ = 2*np.sign(s_)*np.min(s_,s)
        else:
            y_ = p
        return y_
def main():
    f = lambda x: (1/np.tanh(x))
    x = np.arange(0.25,10,0.25)
    y = f(x)
    x_hat = np.arange(0.25,10,0.25)
    y_hat = f(x_hat)

    options = {'kernel':'CS','extrapolate':True}
    interpolator = Interpolator(x, y, options)
    cs = interpolator(x_hat)

    options = {'kernel':'linear','extrapolate':True}
    interpolator = Interpolator(x, y, options)
    linear = interpolator(x_hat)

    options = {'kernel':'MCS','extrapolate':True}
    interpolator = Interpolator(x, y, options)
    mcs = interpolator(x_hat)


    print(pd.DataFrame({'Real':y_hat,'CS':cs,'Linear':linear,'MCS':mcs}))

    plt.figure()
    plt.plot(x_hat,y_hat, label='Real')
    plt.plot(x_hat,cs, label='Cubic Spline')
    plt.plot(x_hat,linear, label='Linear')
    plt.plot(x_hat,mcs, label='Monotonic Cubic Spline')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
