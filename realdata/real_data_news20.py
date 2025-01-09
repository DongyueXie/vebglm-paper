from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

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



def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def load_dataset(categories,verbose=False, seed=42,remove=()):
    """Load and vectorize the 20 newsgroups dataset."""

    data_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=seed,
        remove=remove,
    )

    data_test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        shuffle=True,
        random_state=seed,
        remove=remove,
    )

    # order of labels in `target_names` can be different from `categories`
    target_names = data_train.target_names

    # split target in a training set and a test set
    y_train, y_test = data_train.target, data_test.target

    # Extracting features from the training data using a sparse vectorizer
    t0 = time()
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    X_train = vectorizer.fit_transform(data_train.data)
    duration_train = time() - t0

    # Extracting features from the test data using the same vectorizer
    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    duration_test = time() - t0

    feature_names = vectorizer.get_feature_names_out()

    # remove columns with only zeros in the training set, then the corresponding columns in the test set
    mask = X_train.getnnz(axis=0) > 0
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]

    # remove rows with only zeros in the X_train, and also the X_test
    mask = X_train.getnnz(axis=1) > 0
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = X_test.getnnz(axis=1) > 0
    X_test = X_test[mask]
    y_test = y_test[mask]
    

    if verbose:
        # compute size of loaded data
        data_train_size_mb = size_mb(data_train.data)
        data_test_size_mb = size_mb(data_test.data)

        print(
            f"{len(data_train.data)} documents - "
            f"{data_train_size_mb:.2f}MB (training set)"
        )
        print(f"{len(data_test.data)} documents - {data_test_size_mb:.2f}MB (test set)")
        print(f"{len(target_names)} categories")
        print(
            f"vectorize training done in {duration_train:.3f}s "
            f"at {data_train_size_mb / duration_train:.3f}MB/s"
        )
        print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        print(
            f"vectorize testing done in {duration_test:.3f}s "
            f"at {data_test_size_mb / duration_test:.3f}MB/s"
        )
        print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_names, target_names




categories_all = [
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
#  'misc.forsale',
#  'rec.autos',
#  'rec.motorcycles',
#  'rec.sport.baseball',
#  'rec.sport.hockey',
#  'sci.crypt',
#  'sci.electronics',
#  'sci.med',
#  'sci.space',
#  'talk.politics.guns',
#  'talk.politics.mideast',
#  'talk.politics.misc',
#  'talk.religion.misc',
#  'alt.atheism',
#  'soc.religion.christian',
]

def real_benchmark(models,metrics,file_name=None):
    np.random.seed(42)
    results = []
    
    for i in range(len(categories_all)-1):
        for j in range(i+1,len(categories_all)):
            print(f"running rep {i}-{j}")
            start_t = timeit.default_timer()
            categories = [categories_all[i],categories_all[j]]
            # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=2/5,random_state=i)
            X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
                verbose=True,remove = ('headers', 'footers', 'quotes'),categories=categories
            )
            X_train = X_train.todense()
            X_test = X_test.todense()

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
                            'model': model_name,
                            'metric': metric.__name__,
                            'score': score,
                            'run_time': model.run_time,
                            'categories': categories
                                })
                except Exception as e:
                    print(f"Error with model {model.model_name}: {e}")
                    continue
            end_t = timeit.default_timer()            
            with open(f"realdata/results/{file_name}.pkl", "wb") as fp:
                pickle.dump(results,fp)
            print(f"Rep {i}-{j} took {end_t-start_t} seconds")




def main():

    models = [
        VEBGLM(prior='ash',tol=1e-6,solver='L-BFGS-B2',name_suffix='L-BFGS-B2'),
        VEBGLM(prior='point_normal',tol=1e-6,solver='L-BFGS-B2',name_suffix='L-BFGS-B2'),
        VEBGLM(prior='point_laplace',tol=1e-6,solver='L-BFGS-B2',name_suffix='L-BFGS-B2'),

        VEBGLM(prior='ash',tol=1e-6,solver='L-BFGS-B',name_suffix='L-BFGS-B'),
        VEBGLM(prior='point_normal',tol=1e-6,solver='L-BFGS-B',name_suffix='L-BFGS-B'),
        VEBGLM(prior='point_laplace',tol=1e-6,solver='L-BFGS-B',name_suffix='L-BFGS-B'),
        glmnetR(penalty="lasso"),
        L0Learn(),
        ncvregR(penalty='SCAD'),
        ncvregR(penalty='MCP'),
        varbvsR(),
        ]
    metrics = [accuracy_score, precision_score, recall_score, f1_score,roc_auc_score]
    real_benchmark(models,metrics,file_name='comp')
        

if __name__ == "__main__":
    main()


