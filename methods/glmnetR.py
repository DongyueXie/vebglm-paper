import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import r,numpy2ri, NULL
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import timeit

class glmnetR:
    def __init__(self,penalty="lasso",family="binomial"):
        self.model_name = penalty
        self.family = family

    def fit(self, X,y,weights=None,standardize=False,intercept=True):
        startime = timeit.default_timer()
        X = np.array(X)
        y = np.array(y).squeeze()
        numpy2ri.activate()
        glmnet = importr('glmnet')
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = ro.IntVector(y)
        if self.model_name == 'ridge':
            alpha = 0.0
        elif self.model_name == 'lasso':
            alpha = 1.0
        else:
            raise ValueError('lasso or ridge')
        if weights:
            weights = np.array(weights)
            weights = ro.FloatVector(weights)
        else:
            weights = NULL
        r('set.seed(12345)')
        cv_fit = glmnet.cv_glmnet(x=X_r, y=y_r, family=self.family,weights = weights, standardize=standardize,intercept=intercept,alpha=alpha)
        optimal_lambda = cv_fit.rx2('lambda.1se')
        final_fit = glmnet.glmnet(x=X_r, y=y_r, family=self.family, lambda_=optimal_lambda,weights = weights, standardize=standardize,intercept=intercept,alpha=alpha)

        coefficients = r['as.matrix'](glmnet.coef_glmnet(final_fit, s=optimal_lambda))
        coef_array = np.array(coefficients)[:, 0]  
        intercept = coef_array[0]  
        fitted_coefs = coef_array[1:] 

        self.fitted_model = final_fit
        self.optimal_lambda=optimal_lambda
        self.run_time = timeit.default_timer() - startime
        self.beta = fitted_coefs
        self.intercept = intercept
    def predict_mu(self,X):
        X = np.array(X)
        numpy2ri.activate()
        glmnet = importr('glmnet')
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        mu_pred = glmnet.predict_glmnet(self.fitted_model, newx=X_r, type="link",s = self.optimal_lambda)
        mu_pred = np.array(mu_pred).squeeze()
        return mu_pred
    def predict_proba(self,X):
        if self.family == 'binomial':
            return 1/(1+np.exp(-self.predict_mu(X)))
        else:
            raise ValueError('only for logistics regression')
    def predict(self,X):
        if self.family == 'binomial':
            return (self.predict_proba(X) > 0.5).astype(int).flatten() 
        elif self.family == 'poisson':
            return np.exp(self.predict_mu(X))
        elif self.family == 'gaussian':
            return self.predict_mu(X)


class elasticnetR:
    def __init__(self,family="binomial"):
        self.model_name = 'elasticnet'
        self.family=family

    def fit(self, X,y,weights=None,standardize=False,intercept=True):
        
        startime = timeit.default_timer()
        X = np.array(X)
        y = np.array(y).squeeze()
        numpy2ri.activate()
        glmnet = importr('glmnet')
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        y_r = ro.IntVector(y)
        alpha_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        if weights:
            weights = np.array(weights)
            weights = ro.FloatVector(weights)
        else:
            weights = NULL
        r('set.seed(12345)')
        # best_model = None
        best_model_loss = float('Inf') 
        optimal_lambda = None
        for alpha in alpha_list:
            cv_fit = glmnet.cv_glmnet(x=X_r, y=y_r, family=self.family,weights = weights, standardize=standardize,intercept=intercept,alpha=alpha)
            if min(r['as.vector'](cv_fit.rx2('cvm'))) < best_model_loss:
                # best_model = cv_fit
                best_model_loss = min(r['as.vector'](cv_fit.rx2('cvm')))
                optimal_lambda = cv_fit.rx2('lambda.1se')
            
        final_fit = glmnet.glmnet(x=X_r, y=y_r, family=self.family, lambda_=optimal_lambda,weights = weights, standardize=standardize,intercept=intercept,alpha=alpha)

        coefficients = r['as.matrix'](glmnet.coef_glmnet(final_fit, s=optimal_lambda))
        coef_array = np.array(coefficients)[:, 0]  
        intercept = coef_array[0]  
        fitted_coefs = coef_array[1:] 

        self.fitted_model = final_fit
        self.optimal_lambda=optimal_lambda
        
        self.run_time = timeit.default_timer() - startime
        self.beta = fitted_coefs
        self.intercept = intercept
    def predict_mu(self,X):
        X = np.array(X)
        numpy2ri.activate()
        glmnet = importr('glmnet')
        X_r = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        mu_pred = glmnet.predict_glmnet(self.fitted_model, newx=X_r, type="link",s = self.optimal_lambda)
        mu_pred = np.array(mu_pred).squeeze()
        return mu_pred
    def predict_proba(self,X):
        if self.family == 'binomial':
            return 1/(1+np.exp(-self.predict_mu(X)))
        else:
            raise ValueError('only for logistics regression')
    def predict(self,X):
        if self.family == 'binomial':
            return (self.predict_proba(X) > 0.5).astype(int).flatten() 
        elif self.family == 'poisson':
            return np.exp(self.predict_mu(X))
        elif self.family == 'gaussian':
            return self.predict_mu(X) 


