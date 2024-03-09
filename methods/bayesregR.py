import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
import pandas as pd
import timeit

class bayesregR:
    def __init__(self,prior='hs',family='binomial'):
        self.model_name = "bayesreg"
        self.prior = prior
        self.family = family

    def fit(self, X,y):
        startime = timeit.default_timer()
        X = np.array(X)
        y = np.array(y).squeeze()
        pandas2ri.activate()
        df = pd.DataFrame(X)
        if self.family == 'binomial':
            df['y'] = pd.Categorical(y.astype(str))
        else:
            df['y'] = y
        data_r = pandas2ri.py2rpy(df) 
        ro.r.assign("data_r", data_r)
        if self.family == 'binomial':
            ro.r('data_r$y <- as.factor(data_r$y)')
        bayesreg = importr('bayesreg')
        formula = Formula('y ~ .')
        bayesreg_fit = bayesreg.bayesreg(formula, data=ro.r('data_r'), model=self.family, prior=self.prior)
        self.fitted_model = bayesreg_fit
        self.beta = np.array(self.fitted_model.rx2('mu.beta'))
        self.run_time = timeit.default_timer() - startime
    def predict_mu(self,X):
        X = np.array(X)
        intercept = self.fitted_model.rx2('mu.beta0')
        fitted_coefs = self.fitted_model.rx2('mu.beta')
        mu_pred = np.matmul(X,np.array(fitted_coefs).reshape(-1,1))+np.array(intercept)
        return mu_pred.squeeze()            
    def predict_proba(self,X):
        y_pred_prob = self.sigmoid(self.predict_mu(X))
        return y_pred_prob.squeeze()
    def predict(self,X):
        if self.family == 'binomial':
            return (self.predict_proba(X) > 0.5).astype(int).flatten() 
        elif self.family == 'poisson':
            return np.exp(self.predict_mu(X))
        elif self.family == 'gaussian':
            return self.predict_mu(X)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))