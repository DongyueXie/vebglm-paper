import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import r,numpy2ri,NULL
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import timeit

class sparsevbR:
    def __init__(self,family="logistic"):
        self.model_name = "sparsevb"
        self.family = family

    def fit(self, X,y,intercept=True):
        startime = timeit.default_timer()
        X = np.array(X)
        y = np.array(y).squeeze()
        numpy2ri.activate()
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = ro.IntVector(y)
        sparsevb = importr('sparsevb')
        sparsevb_fit = sparsevb.svb_fit(X=X_r, Y=y_r, family=self.family,slab='laplace',intercept = intercept)
        self.fitted_model = sparsevb_fit
        self.run_time = timeit.default_timer() - startime
    def predict_mu(self,X):
        X = np.array(X)
        sparsevb = importr('sparsevb')
        intercept = self.fitted_model.rx2('intercept')
        fitted_coefs = self.fitted_model.rx2('mu')
        mu_pred = np.matmul(X,np.array(fitted_coefs).reshape(-1,1))+np.array(intercept)
        return mu_pred.squeeze()
    def predict_proba(self,X):
        y_pred_prob = self.sigmoid(self.predict_mu(X))
        return y_pred_prob
    def predict(self,X):
        if self.family == 'logistic':
            return (self.predict_proba(X) > 0.5).astype(int).flatten() 
        elif self.family == 'linear':
            return self.predict_mu(X)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))