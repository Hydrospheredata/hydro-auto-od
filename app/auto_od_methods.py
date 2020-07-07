
from abc import ABC
from typing import Dict
from emmv import em, mv
import numpy as np
import random
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM



class EMMVComparator(ABC):
    """ Compare two EM-MV score pairs.
     Returns if first OD method is better than second"""
    def compare(self, od_method1, od_method2) -> bool:
        pass

class iForest_OCSVM_Comparator(EMMVComparator):
    def compare(self, od_method1, od_method2):
        return od_method1.em > od_method2.em

class LOFComparator(EMMVComparator):
    def compare(self, od_method1, od_method2):
        return od_method1.mv > od_method2.mv


class OutlierDetectionMethod(ABC):
    """ Wrapper class for different outlier detection methods"""
    comparator = EMMVComparator()
    od_method_constructor = None

    def __init__(self, hyperparameters: Dict):
        self.threshold = None
        self.em = em
        self.mv = mv
        self.hyperparameters = hyperparameters
        self.model = None
        self.X = None
        self.ocsvm_max_train = 10000
      
    def evaluate(self, X):
        n_generated = 100000
        _, n_features = np.shape(X)
        lim_inf = X.min(axis=0)
        lim_sup = X.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod() 

        clean_data_percent = 0.9  
        ocsvm_max_train = 10000   
        alpha_min = 0.9
        alpha_max = 0.999
        x_axis = np.arange(0, 100, 0.01) 
        axis_alpha = np.arange(alpha_min, alpha_max, 0.0001) 
        n_generated = 100000

        uniform_samples = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features)) 
        clean_data_score = self.model.decision_function(X)
        dirty_data_score = self.model.decision_function(uniform_samples)

        # TODO calculate EM-MV
        self.em, _, _ = em(x_axis, clean_data_percent, volume_support, dirty_data_score, clean_data_score, n_generated)
        self.mv, _ = mv(axis_alpha, volume_support, dirty_data_score, clean_data_score, n_generated)
        del self.model  # Clean memory

        return self.em, self.mv

    def decision_function(self, X):
        return self.model.decision_function(X)

    def __gt__(self, other):
        self.comparator.compare(self, other)


class AutoIForest(OutlierDetectionMethod):
    od_method_constructor = IsolationForest
    comparator = iForest_OCSVM_Comparator()


    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoIForest.od_method_constructor(**hyperparameters)

    def fit(self, X):
        self.model.fit(X)
        self.model.threshold = self.model.offset_
        

class AutoLOF(OutlierDetectionMethod):
    od_method_constructor = LocalOutlierFactor
    comparator = LOFComparator()

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoLOF.od_method_constructor(**hyperparameters)

    def fit(self, X):
        self.model.fit(X)
        self.model.threshold = self.model.offset_

class AutoOCSVM(OutlierDetectionMethod):
    od_method_constructor = OneClassSVM
    comparator = iForest_OCSVM_Comparator()

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoOCSVM.od_method_constructor(**hyperparameters)

    def fit(self, X):
        if len(X) > self.ocsvm_max_train:
          self.model.fit(random.choices(X, k = ocsvm_max_train))
          self.threshold = self.model.offset_
        else:
          self.model.fit(X)
          self.threshold = self.model.offset_
          



