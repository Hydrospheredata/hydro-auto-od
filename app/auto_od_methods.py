
from abc import ABC
from typing import Dict
from emmv import em, mv
import logging
import numpy as np
import random
from hydrosdk.monitoring import TresholdCmpOp
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM



class OutlierDetectionMethod(ABC):
    """ Wrapper class for different outlier detection methods"""
    od_method_constructor = None

    def __init__(self, hyperparameters: Dict):
        self.em = None
        self.mv = None
        self.hyperparameters = hyperparameters
        self.model = None
        self.ocsvm_max_train = 10000
      
    def evaluate(self, X, x_test):
        n_generated = 100000
        _, n_features = np.shape(X)
        lim_inf = X.min(axis=0)
        lim_sup = X.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod() 
        if volume_support == np.inf:
            logging.info('Volume Support is inf. Metric might be biased.')
        if volume_support == 0.0:
            logging.info('Volume Support is 0.0. Check your data format.')
        clean_data_percent = 0.9  
        alpha_min = 0.9
        alpha_max = 0.999
        axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
        x_axis = np.arange(0, 100, 0.01) 

        uniform_samples = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features)) 
        clean_data_score = self.model.decision_function(x_test)
        dirty_data_score = self.model.decision_function(uniform_samples)

        # calculate EM-MV
        self.em, _, _ = em(x_axis, clean_data_percent, volume_support, dirty_data_score, clean_data_score, n_generated)
        self.mv, _ = mv(axis_alpha, volume_support, dirty_data_score, clean_data_score, n_generated)
        del self.model  # Clean memory

    def decision_function(self, X):
        return self.model.decision_function(X)

    def recreate(self, X):
        self.model = self.od_method_constructor(**self.hyperparameters).fit(X)
        self.model.threshold_ = self.model.offset_
        return self.model


class AutoIForest(OutlierDetectionMethod):
    od_method_constructor = IsolationForest
    threshold_comparator = TresholdCmpOp.GREATER

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoIForest.od_method_constructor(**hyperparameters)

    def fit(self, X):
        self.model.fit(X)
        

class AutoLOF(OutlierDetectionMethod):
    od_method_constructor = LocalOutlierFactor
    threshold_comparator = TresholdCmpOp.GREATER

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoLOF.od_method_constructor(**hyperparameters)

    def fit(self, X):
        self.model.fit(X)


class AutoOCSVM(OutlierDetectionMethod):
    od_method_constructor = OneClassSVM
    threshold_comparator = TresholdCmpOp.GREATER

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoOCSVM.od_method_constructor(**hyperparameters)

    def fit(self, X):
        if len(X) > self.ocsvm_max_train:
            self.model.fit(random.choices(np.array(X), k=self.ocsvm_max_train))
        else:
            self.model.fit(X)



          



