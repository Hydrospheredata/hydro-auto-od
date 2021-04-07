import numpy as np
from sklearn.metrics import auc
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.utils import shuffle as sh
from sklearn.model_selection import ParameterGrid, train_test_split


def compute_mv(clf, X_train, X_test, alphas, 
               lim_inf, lim_sup, n_sim, features, vol_sup):
    # Random Uniform sampling
    U = np.random.uniform(lim_inf, lim_sup, 
                size=[n_sim, len(features)])
    # Training classifier
    clf = clf.fit(X_train)
    score_U = -clf.decision_function(U)
    score_test = -clf.decision_function(X_test)
    # compute offsets
    offsets_p = np.percentile(score_test, 100 * (1 - alphas))
    # compute volumes of associated level sets
    vol_p = (np.array([np.mean(score_U >= offset) for offset in offsets_p]) *
             vol_sup)
    return vol_p


def low_tuning(X_train, X_test, object_list, base_estimator = None, 
               alphas=np.arange(0.05, 1., 0.05), n_sim = 100000):
    max_features = 5
    _, n_features = X_train.shape
    features_list = np.arange(n_features)
    auc_test = np.zeros(len(object_list))
    combs = combinations(features_list, max_features)
    for comb in combs:
        X_train_ = X_train[:, comb]
        X_ = X_test[:, comb]
        lim_inf = X_.min(axis=0)
        lim_sup = X_.max(axis=0)
        for p, object_ in enumerate(object_list):
            volume_support = (lim_sup - lim_inf).prod()
            if volume_support > 0:
                if base_estimator is None:
                    clf = object_()
                else:
                    clf = base_estimator(**object_)
                vol_p = compute_mv(clf, X_train_, X_, alphas, lim_inf, 
                           lim_sup, n_sim, comb, volume_support)
                auc_test[p] = auc(alphas, vol_p)
    auc_test /= len(combs)
    best_p = np.argmin(auc_test)
    best_ = object_list[best_p]
    return best_
    

def high_tuning(X_train, X_test, object_list, base_estimator = None, 
               alphas=np.arange(0.05, 1., 0.05), averaging = 50, n_sim = 100000):
    max_features = 5
    _, n_features = X_train.shape
    auc_est = np.zeros(len(object_list))
    for p, object_ in enumerate(object_list):
        nb_exp = 0
        while nb_exp < averaging:  
            features = sh(np.arange(n_features))[:max_features]
            X_train_ = X_train[:, features]
            X_ = X_test[:, features]
            lim_inf = X_.min(axis=0)
            lim_sup = X_.max(axis=0)
            volume_support = (lim_sup - lim_inf).prod()
            if volume_support > 0:
                nb_exp += 1
                if base_estimator is None:
                    clf = object_()
                else:
                    clf = base_estimator(**object_)
                vol_p = compute_mv(clf, X_train_, X_, alphas, lim_inf, 
                                   lim_sup, n_sim, features, volume_support)
                auc_est[p] += auc(alphas, vol_p)
    auc_est /= averaging 
    best_p = np.argmin(auc_est)
    best_ = object_list[best_p]
    return best_


def model_tuning(X_train, X_test, base_estimator=None, parameters=None, alphas=np.arange(0.05, 1., 0.05)):
    param_grid = ParameterGrid(parameters)
    _, n_features = X_train.shape
    if n_features <= 7:
        res = low_tuning(X_train, X_test, base_estimator=base_estimator, object_list = param_grid, alphas=alphas, n_sim = 10000) 
    else:
        res = high_tuning(X_train, X_test, base_estimator=base_estimator, object_list = param_grid, averaging = 10, alphas=alphas, n_sim = 10000) 
    return res

