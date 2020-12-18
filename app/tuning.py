import numpy as np
from sklearn.utils import shuffle as sh
from sklearn.metrics import auc
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, train_test_split



def compute_mv(score_U, score_test, alphas, vol_sup):
    
    # compute offsets
    offsets_p = np.percentile(score_test, 100 * (1 - alphas))
    # compute volumes of associated level sets
    vol_p = (np.array([np.mean(score_U >= offset) for offset in offsets_p]) *
             vol_sup)

    return vol_p, offsets_p


def est_tuning(X_train, X_test, object_list, base_estimator = None, 
               alphas=np.arange(0.05, 1., 0.05), averaging = 50, n_sim = 10000):
    
    max_features = 5
    _, n_features = X_train.shape
    if n_features < max_features:
        max_features = n_features
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
                U = np.random.uniform(lim_inf, lim_sup, 
                            size=[n_sim, len(features)])
                if base_estimator is None:
                    clf = object_
                else:
                    clf = base_estimator(**object_)
                clf = clf.fit(X_train_)
                score_U = -clf.decision_function(U)
                score_test = -clf.decision_function(X_)
                vol_p, _ = compute_mv(score_U, score_test, alphas,
                                              volume_support)
                auc_est[p] += auc(alphas, vol_p)

    auc_est /= averaging 
    best_p = np.argmin(auc_est)
    best_ = object_list[best_p]
    return best_


def model_tuning(X, base_estimator=None, parameters=None,
                 cv=None, alphas=np.arange(0.05, 1., 0.05), n_jobs=-1):
    

    param_grid = ParameterGrid(parameters)
    res = Parallel(n_jobs=n_jobs, verbose=10)(delayed(est_tuning)(X[train], X[test], base_estimator=base_estimator, object_list = param_grid, alphas=alphas, averaging = 10) for train, test in cv.split(X))
    
    return res
