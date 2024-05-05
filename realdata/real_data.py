import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.realpath(__file__))
myPy_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(os.path.join(myPy_dir, 'VEBGLM', 'src'))

from VEBGLM import VEBGLM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from VEBGLM.src.VEBGLM import VEBGLM
from methods.bayesregR import bayesregR
from methods.glmnetR import glmnetR,elasticnetR
from methods.L0Learn import L0Learn
from methods.ncvregR import ncvregR
from methods.sparsevbR import sparsevbR
from methods.varbvsR import varbvsR
import argparse
import pickle
import timeit
from process_data import *


def real_benchmark(X,y,models,metrics,reps=20,data_name=None,file_name=None):
    results = []
    for i in range(reps):
        print(f"running data {data_name}, rep {i}")
        start_t = timeit.default_timer()
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=2/5,random_state=i)
        for model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                model_name =  model.model_name
                for metric in metrics:
                    if metric.__name__ == 'roc_auc_score':
                        score = metric(y_test, y_prob)
                    else:    
                        score = metric(y_test, y_pred)
                    results.append({
                        'rep': i,
                        'model': model_name,
                        'metric': metric.__name__,
                        'score': score,
                        'run_time': model.run_time,
                        'data_name': data_name,
                        'seed': i
                            })
            except Exception as e:
                print(f"Error with model {model.model_name}: {e}")
                continue
        end_t = timeit.default_timer()            
        with open(f"realdata/results/{data_name}-{file_name}.pkl", "wb") as fp:
            pickle.dump(results,fp)
        print(f"Rep {i} took {end_t-start_t} seconds")


def main():

    models = [
        VEBGLM(prior='ash',tol=1e-6,solver='L-BFGS-B2'),
        VEBGLM(prior='point_normal',tol=1e-6,solver='L-BFGS-B2'),
        VEBGLM(prior='point_laplace',tol=1e-6,solver='L-BFGS-B2'),
        elasticnetR(),
        glmnetR(penalty="lasso"),
        L0Learn(),
        ncvregR(penalty='SCAD'),
        ncvregR(penalty='MCP'),
        varbvsR(),
        ]
    metrics = [accuracy_score, precision_score, recall_score, f1_score,roc_auc_score]
    datasets = [Ionosphere, annealing, adult, heart_disease, Pediatric, abalone]
    # datasets = [musk]

    for data in datasets:
        X,y = data()
        real_benchmark(X,y,models,metrics,reps=20,data_name=data.__name__,file_name='final')

if __name__ == "__main__":
    main()