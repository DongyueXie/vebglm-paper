import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import r,numpy2ri,NULL
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import timeit

class varbvsR:
    def __init__(self,family="binomial"):
        self.model_name = "varbvs"
        self.family = family

    def fit(self, X,y,weights=None):
        startime = timeit.default_timer()
        X = np.array(X)
        y = np.array(y).squeeze()
        numpy2ri.activate()
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = ro.IntVector(y)
        varbvs = importr('varbvs')
        if weights:
            weights = np.array(weights)
            weights = ro.FloatVector(weights)
        else:
            weights = NULL
        varbvs_fit = varbvs.varbvs(X=X_r, y=y_r, Z= ro.NULL, family=self.family,weights=weights,verbose=False)

        coef_matrix = r['coef.varbvs'](varbvs_fit)
        coefficients = coef_matrix[:,coef_matrix.shape[1]-1]
        intercept = coefficients[0]
        fitted_coefs = coefficients[1:]
        self.beta = fitted_coefs
        self.intercept = intercept
        
        self.fitted_model = varbvs_fit
        self.run_time = timeit.default_timer() - startime
    def predict_mu(self,X):
        X = np.array(X)
        numpy2ri.activate()
        varbvs = importr('varbvs')
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        mu_pred = varbvs.predict_varbvs(self.fitted_model, X=X_r, type="link")
        return np.array(mu_pred).squeeze()
    def predict_proba(self,X):
        X = np.array(X)
        numpy2ri.activate()
        varbvs = importr('varbvs')
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_pred_prob = varbvs.predict_varbvs(self.fitted_model, X=X_r, type="response")
        y_pred_prob_np = np.array(y_pred_prob)  # Convert to numpy array
        return y_pred_prob_np.squeeze()
    def predict(self,X):
        if self.family == 'binomial':
            return (self.predict_proba(X) > 0.5).astype(int).flatten() 
        elif self.family == 'gaussian':
            return self.predict_mu(X)