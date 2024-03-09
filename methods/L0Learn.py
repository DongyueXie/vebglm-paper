import l0learn
import numpy as np
import timeit

class L0Learn:
    def __init__(self,family = 'Logistic'):
        self.model_name = "L0Learn"
        self.family = family
    def fit(self,X,y):
        startime = timeit.default_timer()
        fit_model = l0learn.cvfit(np.array(X,dtype=np.float64),np.array(y.squeeze(),dtype=np.float64),loss=self.family,penalty='L0')
        self.fitted_model = fit_model
        self.optimal_lambda = fit_model.lambda_0[0][np.argmin(fit_model.cv_means)]
        self.optimal_beta = fit_model.coeffs[0].toarray()[:,np.argmin(fit_model.cv_means)]
        self.run_time = timeit.default_timer() - startime  
    def predict_proba(self,X):
        y_predict_prob = self.fitted_model.predict(np.array(X,dtype=np.float64),lambda_0=self.optimal_lambda,gamma=self.fitted_model.gamma)
        return y_predict_prob.squeeze()
    def predict(self, X):
        if self.family == 'Logistic':
            return (self.predict_proba(X) > 0.5).astype(int).flatten() 
        elif self.family == 'SquaredError':
            return self.fitted_model.predict(np.array(X,dtype=np.float64),lambda_0=self.optimal_lambda,gamma=self.fitted_model.gamma).squeeze()