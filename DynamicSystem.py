import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


class StateSpaceLearner(BaseEstimator, TransformerMixin):
    def __init__(self, time_delay = 0, fill_in = 0):
        self.time_delay = -abs(time_delay)
        self.fill_in = fill_in
        
    def fit(self, X, y, **kwargs):
        
        x_prev = np.roll(y, self.time_delay, axis = 0)
        _, self.y_s = y.shape
        
        x_prev[self.time_delay:,-1] = np.array([self.fill_in]*-self.time_delay)
        print(x_prev[self.time_delay,-1])
        #x_prev[0:self.time_delay,-1] = np.zeros(self.time_delay)
        
        X = np.c_[X, x_prev]
        _, self.n_features_in_ = X.shape
        self.lr = LinearRegression()
        self.lr.fit(X,y,**kwargs)
        self.coef_ = lr.coef_
        self.intercept_ = lr.intercept_
        return self

    def predict(self, X):
        
        y_size, x_size= X.shape
        y = np.zeros((y_size, x_size))
        X_t = np.c_[X, y]
        print(X_t.shape)
        
        for idx, obs in enumerate(X_t):
            if (idx + 1) >= y_size:
                return X_t[:,-self.y_s:]
            X_t[idx+1, -self.y_s:] = self.lr.predict(obs.reshape(1,-1))
        
class RecursiveLearner(BaseEstimator, TransformerMixin):
    def __init__(self, time_delay = 0, fill_in = 0):
        self.time_delay = -abs(time_delay)
        self.fill_in = fill_in
        
    def fit(self, X, y, **kwargs):
        
        x_next = np.roll(y, self.time_delay, axis = 0)
        y_next = np.roll(x_next, -1, axis = 0)
        _, self.y_s = y.shape
        
        x_next[self.time_delay:,-self.y_s:] = np.array([self.fill_in]*-self.time_delay*self.y_s).reshape(-self.time_delay,self.y_s)
      
        #x_prev[0:self.time_delay,-1] = np.zeros(self.time_delay)

        
        #X = np.c_[X, x_prev]
        _, self.n_features_in_ = X.shape
        self.lr = LinearRegression()
        self.lr.fit(X,x_next,**kwargs)
        self.coef_ = self.lr.coef_
        self.intercept_ = self.lr.intercept_
        return self
    
    def predict(self, X):
        
        y_size, x_size= X.shape
        y = np.zeros((y_size, x_size))
        X_t = np.c_[X, y]
        print(X_t.shape)
        
        y = self.lr.predict(X)
        
        for idx, obs in enumerate(X):
            if (idx + 1) >= y_size:
                return y #X_t[:,-self.y_s:]
            y[idx] = self.lr.predict(obs.reshape(1,-1))
            

            
