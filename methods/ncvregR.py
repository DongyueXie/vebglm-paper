import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import r,numpy2ri,NULL
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import timeit

class ncvregR:
    def __init__(self,penalty='SCAD',family="binomial"):
        self.model_name = penalty
        self.family = family

    def fit(self, X,y):
        startime = timeit.default_timer()
        X = np.array(X)
        y = np.array(y).squeeze()
        numpy2ri.activate()
        ncvreg = importr('ncvreg')
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = ro.IntVector(y)
        r('set.seed(12345)')
        cv_fit = ncvreg.cv_ncvreg(X=X_r, y=y_r, family=self.family,penalty=self.model_name,warn=False)
        coefficients = cv_fit.rx2('fit').rx2('beta')[:,int(cv_fit.rx2('min')[0])-1]
        coef_array = coefficients.squeeze()  # Assuming optimal_lambda results in a single column of coefficients
        intercept = coef_array[0]  # Intercept
        fitted_coefs = coef_array[1:] 
        self.beta = fitted_coefs
        self.intercept = intercept
        self.fitted_model = cv_fit
        self.run_time = timeit.default_timer() - startime
    def predict_mu(self,X):
        mu_pred = np.array(np.matmul(X,np.array(self.beta).reshape(-1,1))+np.array(self.intercept)).squeeze()
        return mu_pred
    def predict_proba(self,X):
        X = np.array(X)
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

