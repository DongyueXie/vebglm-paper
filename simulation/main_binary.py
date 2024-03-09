import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/dxie/myPy/VEBGLM/src')
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
from utils_func import generate_train_data_binary,generate_test_data_binary,gen_beta,X_generator




def main(args):
    np.random.seed(12345)
    n_values = [int(n) for n in args.n_values.split(',')]
    p_values = [int(p) for p in args.p_values.split(',')]
    r_values = [float(r) for r in args.r_values.split(',')]
    s_values = [int(s) for s in args.s_values.split(',')]
    dist_values = [str(d) for d in args.beta_dist.split(',')]
    repetitions = args.repetitions
    # target_var = args.target_var
    # Define models and metrics
    # bayesregR(),
    models = [VEBGLM(prior='ash'),VEBGLM(prior='point_normal'),VEBGLM(prior='point_laplace'),
              VEBGLM(prior='ash'),VEBGLM(prior='point_normal'),VEBGLM(prior='point_laplace'),
              VEBGLM(prior='ash'),VEBGLM(prior='point_normal'),VEBGLM(prior='point_laplace'),
              VEBGLM(prior='ash'),VEBGLM(prior='point_normal'),VEBGLM(prior='point_laplace'),
              ] 
            #   glmnetR(penalty="lasso"),elasticnetR(),
            #   glmnetR(penalty="ridge"),
            #   L0Learn(),
            #   ncvregR(penalty='SCAD'),ncvregR(penalty='MCP'),
            #   sparsevbR(),varbvsR(),bayesregR(prior='hs')
    
              
    metrics = [accuracy_score, precision_score, recall_score, f1_score,roc_auc_score]

    # Run simulation
    results = []
    datax = []
    for n in n_values:
        for p in p_values:
            for s in s_values:
                for r in r_values:
                    for rep in range(repetitions):
                        for dist in dist_values:
                            print(f"running setting {n,p,s,r,dist}, rep {rep}")
                            # beta = gen_beta_constant(n,p,r,s,target_var)
                            beta = gen_beta(p,s,dist)
                            X_train, y_train = generate_train_data_binary(n, p, r, beta, X_generator)
                            X_test, y_test = generate_test_data_binary(args.n_test, p, r,beta, X_generator)
                            datax.append({
                                        'n': n,
                                        'p': p,
                                        'r': r,
                                        's': s,
                                        'beta_dist': dist,
                                        'rep': rep,
                                        'beta':beta,
                                        'X_train':X_train,
                                        'y_train':y_train,
                                        'X_test':X_test,
                                        'y_test':y_test})
                            # with open(f"simulation/results/simu_data_{args.file_name}.pkl",'wb') as aa:
                            #     pickle.dump(datax, aa)
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
                                            'n': n,
                                            'p': p,
                                            'r': r,
                                            's': s,
                                            'beta_dist': dist,
                                            'rep': rep,
                                            'model': model_name,
                                            'metric': metric.__name__,
                                            'score': score,
                                            'run_time': model.run_time
                                        })
                                    # save result after every modelling fitting
                                    with open(f"simulation/results/simu_res_{args.file_name}.pkl", "wb") as fp:
                                        pickle.dump(results, fp)
                                except Exception as e:
                                    print(f"Error with model {model.model_name}: {e}")
                                    continue



    return datax, results


    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run logistic regression simulation.')
    parser.add_argument('--n_values', type=str, default='500,1000', help='Comma-separated list of sample sizes')
    parser.add_argument('--p_values', type=str, default='20,50,100,500,1000,3000', help='Comma-separated list of dimensions')
    parser.add_argument('--s_values', type=str, default='1,2,4,8,16,20', help='Comma-separated list of sparsity levels')
    parser.add_argument('--r_values', type=str, default='0.0,0.1,0.3,0.5,0.7,0.9', help='Comma-separated list of sparsity levels')
    parser.add_argument('--repetitions', type=int, default=20, help='Number of repetitions for each setting')
    parser.add_argument('--target_var', type=float, default=5.0, help='Number of repetitions for each setting')
    parser.add_argument('--n_test', type=int, default=3000, help='Number of repetitions for each setting')
    parser.add_argument('--file_name', type=str, default='default', help='Number of repetitions for each setting')
    parser.add_argument('--beta_dist', type=str, default='normal', help='Number of repetitions for each setting')
    args = parser.parse_args()
    datax, results = main(args)
    # Analyze results
    results_df = pd.DataFrame(results)
    res = results_df.groupby(['model', 'metric','n','p','r','s','beta_dist']).mean().reset_index()
    print(res[res['metric']=='roc_auc_score'])