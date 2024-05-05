# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.linalg import toeplitz
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# Define data generation methods
def generate_train_data_binary(n, p, r, beta, X_generator):
    X = X_generator(n, p, r)
    y = np.random.binomial(1, 1 / (1 + np.exp(-X @ beta)))
    return X, y

def generate_test_data_binary(n_test,p,r,beta,X_generator):
    X = X_generator(n_test, p, r,scale_by_n=False)
    y = np.random.binomial(1, 1 / (1 + np.exp(-X @ beta)))
    return X,y

def X_generator(n, p, r, scale_by_n=False):
    if r == 0.0:
        if scale_by_n:
            return np.random.normal(0, np.sqrt(1/n), (n, p))
        else:
            return np.random.normal(0, 1, (n, p))
    else:
        if scale_by_n:
            return np.random.multivariate_normal(np.zeros(p), gen_cov(p,r)/n, n)
        else:
            return np.random.multivariate_normal(np.zeros(p), gen_cov(p,r), n)

# def gen_cov(p,r):
#     res = np.zeros((p,p))
#     for i in range(p):
#         for j in range(p):
#             res[i,j] = r**np.abs(j-i)
#     return res

def gen_cov(p, r):
    # Generate the first row of the covariance matrix
    first_row = np.array([r ** i for i in range(p)])
    # Use the first row to generate the entire Toeplitz matrix
    res = toeplitz(first_row)
    return res

def gen_beta(p,s,dist='normal'):
    if dist=='normal':
        return gen_beta_normal(p,s)
    if dist == "laplace":
        return gen_beta_laplace(p,s)
    if dist == "uniform":
        return gen_beta_uniform(p,s)
    if dist == "t":
        return gen_beta_t(p,s)
    if dist == "const":
        return gen_beta_constant(p,s)

def gen_beta_normal(p,s):
    beta = np.zeros((p,1))
    beta[0:s,:] = np.random.randn(s,1)
    return beta
def gen_beta_laplace(p,s):
    beta = np.zeros((p,1))
    beta[0:s,:] = np.random.laplace(size=(s,1))
    return beta
def gen_beta_uniform(p,s):
    beta = np.zeros((p,1))
    beta[0:s,:] = np.random.uniform(-1,1,size=(s,1))
    return beta
def gen_beta_t(p,s,df=1):
    beta = np.zeros((p,1))
    beta[0:s,:] = np.random.standard_t(df=df,size=(s,1))
    return beta

def gen_beta_constant(p,s,const=1.0):
    beta = np.zeros((p,1))
    beta[0:int(s),:] = const
    return beta


# def gen_beta_constant(n,p,r,s,const=1,target_var = None,sign='-'):
#     """generate beta such that var(Xb) = 5"""
#     X = X_generator(n, p, r)
#     beta = np.zeros((p,1))
#     if sign=="+":
#         beta[0:int(s),:] = const
#     elif sign=="-":
#         beta[0:int(s),:] = -const
#     else:
#         beta[0:int(s/2),:] = const
#         beta[int(s/2):s,:] = -const
#     if target_var:
#         beta = beta * np.sqrt(target_var / np.var(X@beta))
#     return beta



# def beta_generator(p, s):
#     beta = np.zeros(p)
#     non_zero_indices = np.random.choice(p, int(p * s), replace=False)
#     beta[non_zero_indices] = np.random.normal(0, 1, int(p * s))
#     return beta