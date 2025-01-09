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
from utils_func import generate_train_data_binary,generate_test_data_binary,gen_beta,X_generator
import timeit



def main(args):
    
    n_values = [int(n) for n in args.n_values.split(',')]
    p_values = [int(p) for p in args.p_values.split(',')]
    r_values = [float(r) for r in args.r_values.split(',')]
    s_values = [int(s) for s in args.s_values.split(',')]
    dist_values = [str(d) for d in args.beta_dist.split(',')]
    repetitions = args.repetitions
    
    # Use solver = "L-BFGS-B2" and init = 'lasso.1se' for simulations for vebglm

    models = [
            VEBGLM(prior='ash',tol=1e-6,solver='L-BFGS-B2'),
            VEBGLM(prior='point_normal',tol=1e-6,solver='L-BFGS-B2'),
            VEBGLM(prior='point_laplace',tol=1e-6,solver='L-BFGS-B2'),
            elasticnetR(),
            glmnetR(penalty="lasso"),
            glmnetR(penalty="ridge"),
            L0Learn(),
            ncvregR(penalty='SCAD'),
            ncvregR(penalty='MCP'),
            sparsevbR(),
            varbvsR(),
        ]
    
              
    metrics = [accuracy_score, precision_score, recall_score, f1_score,roc_auc_score]

    # Run simulation
    results = []
    datax = []
    fitted_models = []
    for n in n_values:
        for p in p_values:
            for s in s_values:
                for r in r_values:
                    for dist in dist_values:
                        seed = 0
                        for rep in range(repetitions):
                            
                            print(f"running setting {n,p,s,r,dist}, rep {rep}")
                            start_t = timeit.default_timer()
                            np.random.seed(seed)
                            beta = gen_beta(p,s,dist)
                            X_train, y_train = generate_train_data_binary(n, p, r, beta, X_generator)
                            X_test, y_test = generate_test_data_binary(args.n_test, p, r,beta, X_generator)
                            # datax.append({
                            #             'n': n,
                            #             'p': p,
                            #             'r': r,
                            #             's': s,
                            #             'beta_dist': dist,
                            #             'rep': rep,
                            #             'beta':beta,
                            #             'X_train':X_train,
                            #             'y_train':y_train,
                            #             'X_test':X_test,
                            #             'y_test':y_test})
                            # with open(f"simulation/results/simu_data_{args.file_name}.pkl",'wb') as aa:
                            #     pickle.dump(datax, aa)
                            for model in models:
                                try:
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    y_prob = model.predict_proba(X_test)
                                    model_name =  model.model_name
                                    fitted_models.append({
                                        'n': n,
                                        'p': p,
                                        'r': r,
                                        's': s,
                                        'beta_dist': dist,
                                        'rep': rep,
                                        'seed': seed,
                                        'true_beta': beta,
                                        'fitted_model': model,
                                    })

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
                                            'seed': seed,
                                            'model': model_name,
                                            'metric': metric.__name__,
                                            'score': score,
                                            'run_time': model.run_time
                                        })

                                except Exception as e:
                                    print(f"Error with model {model.model_name}: {e}")
                                    continue
                            end_t = timeit.default_timer()
                            seed += 1
                            print(f"Rep {rep} took {end_t-start_t} seconds")
                            with open(f"simulation/results/simu_fitted_model_{args.file_name}.pkl", "wb") as fp:
                                pickle.dump(fitted_models, fp)
                            with open(f"simulation/results/simu_metric_{args.file_name}.pkl", "wb") as fp:
                                pickle.dump(results, fp)


    return results


    



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
    results = main(args)
    # Analyze results
    results_df = pd.DataFrame(results)
    res = results_df.groupby(['model', 'metric','n','p','r','s','beta_dist']).mean().reset_index()
    print(res[res['metric']=='roc_auc_score'])