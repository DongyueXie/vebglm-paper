from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def musk():
    datax = fetch_ucirepo(id=74)
    X = datax.data.features
    y = datax.data.targets
    X = X.drop(columns=['molecule_name', 'conformation_name'])
    y = y['class']
    y = y.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X[:,np.where(X.sum(axis=0)!=0.0)[0].squeeze()]
    return X, y

def Ionosphere():
    """351, 34"""
    datax = fetch_ucirepo(id=52)
    X = datax.data.features
    y = datax.data.targets
    y = y['Class']
    y = y.map({"g": 0, "b": 1})
    y = y.to_numpy()
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X[:,np.where(X.sum(axis=0)!=0.0)[0].squeeze()]
    return X, y

def bankcrupt():
    """6819, 90"""
    datax = fetch_ucirepo(id=572)
    X = datax.data.features
    y = datax.data.targets
    y = y['Bankrupt?']
    y = y.to_numpy()
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X[:,np.where(X.sum(axis=0)!=0.0)[0].squeeze()]
    return X, y

def mushroom():
    """8124, 95"""
    datax = fetch_ucirepo(id=73)
    X = datax.data.features
    y = datax.data.targets
    y = y['poisonous']
    y = y.map({"e": 0, "p": 1})
    y = y.to_numpy()
    numerical_columns = X.select_dtypes(include=['int64','float64']).columns.tolist()
    X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].median())
    cat_columns = X.select_dtypes(include=['object']).columns.tolist()
    X[cat_columns] = X[cat_columns].fillna('missing')
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X[:,np.where(X.sum(axis=0)!=0.0)[0].squeeze()]
    return X, y

def annealing():
    """898, 46"""
    datax = fetch_ucirepo(id=3)
    X = datax.data.features
    y = datax.data.targets
    y = y['class']
    y = y.map({"3": 1, "2": 0, "5": 0, 'U': 0, "1": 0})
    y = y.to_numpy()
    numerical_columns = X.select_dtypes(include=['int64','float64']).columns.tolist()
    X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].median())
    cat_columns = X.select_dtypes(include=['object']).columns.tolist()
    X[cat_columns] = X[cat_columns].fillna('missing')
    X = X.drop(columns=['m','marvi','corr','jurofm','s','p'])
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# def toxicity():
#     """171, 1203
#     Too hard for logistic regression"""
#     datax = fetch_ucirepo(id=728)
#     X = datax.data.features
#     y = datax.data.targets
#     X = pd.get_dummies(X, drop_first=True)
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     y = y.Class
#     y = y.map({'Toxic': 1, 'NonToxic': 0})
#     y = y.to_numpy()
#     X = X[:,np.where(X.sum(axis=0)!=0.0)[0].squeeze()]
#     return X, y
    
def adult():
    """48842 by 103
    fill missing values with mode for category variables
    one-hot encode the category variables
    scale the data"""
    datax = fetch_ucirepo(id=2)
    X = datax.data.features
    y = datax.data.targets
    cat_col = X.select_dtypes(include='object').columns
    X[cat_col] = X[cat_col].fillna('missing')
    y['income'][y['income'] == '<=50K.'] = '<=50K'
    y['income'][y['income'] == '>50K.'] = '>50K'
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.income
    y = y.map({'<=50K': 0, '>50K': 1})
    y = y.to_numpy()
    return X, y

def heart_disease():
    """297, 18. dropped missing data"""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    y.loc[y['num'] != 0, 'num'] = 1
    X = X.dropna()
    y = y.loc[X.index]
    columns_to_encode = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for col in columns_to_encode:
        X[col] = X[col].astype('category')
    X = pd.get_dummies(X, columns=columns_to_encode,drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = y.to_numpy()
    return X_scaled, y

def Pediatric():
    """782, 147 median fill for missing numerical values; add a new category called missing for category varaible"""
    datax = fetch_ucirepo(id=938)
    X = datax.data.features
    y = datax.data.targets
    numerical_columns = X.select_dtypes(include=['int64','float64']).columns
    X[numerical_columns] = X[numerical_columns].apply(lambda col: col.fillna(col.median()))
    cat_col = X.select_dtypes('object').columns
    X[cat_col] = X[cat_col].fillna('missing')
    X = pd.get_dummies(X,drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.Diagnosis
    y = y.fillna('no appendicitis')
    y = y.map({'appendicitis': 1, 'no appendicitis': 0})
    y = y.to_numpy()
    X = X[:,np.where(X.sum(axis=0)!=0.0)[0].squeeze()]
    return X,y

def abalone():
    """4177 by 9
    Fill missing values with median for numerical columns and add a new category called missing for category varaible,
    then one-hot encode the category variables, and scale the data;
    target is the number of rings, convert to binary target by setting the threshold at 9"""
    datax = fetch_ucirepo(id=1)
    X = datax.data.features
    y = datax.data.targets
    numerical_columns = X.select_dtypes(include=['int64','float64']).columns
    X[numerical_columns] = X[numerical_columns].apply(lambda col: col.fillna(col.median()))
    cat_col = X.select_dtypes('object').columns
    X[cat_col] = X[cat_col].fillna('missing')
    X = pd.get_dummies(X,drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = (y>9).astype(int)
    y = y.to_numpy()
    return X,y
