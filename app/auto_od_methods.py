from abc import ABC
from typing import Dict
from emmv import mv
import numpy as np
import random
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

class OutlierDetectionMethod(ABC):
    """ Wrapper class for different outlier detection methods"""
    od_method_constructor = None

    def __init__(self, hyperparameters: Dict):
        self.em = None
        self.mv = None
        self.hyperparameters = hyperparameters
        self.model = None
      
    def evaluate(self, X, x_test):
        # Parameters for mv calculation
        n_generated = 100000
        _, n_features = np.shape(X)
        lim_inf = X.min(axis=0)
        lim_sup = X.max(axis=0)
        volume_support = sum(np.log1p(x) for x in lim_sup - lim_inf)
        axis_alpha = np.arange(0.9, 0.999, 0.0001)
        # Preparing uniformally distributed samples and making predictions for each group of samples
        uniform_samples = np.random.uniform(lim_inf, lim_sup, size=(n_generated, n_features)) 
        clean_data_score = -self.model.decision_function(x_test) 
        dirty_data_score = -self.model.decision_function(uniform_samples) 
        # Calculating MV
        self.mv, _ = mv(axis_alpha, volume_support, dirty_data_score, clean_data_score, n_generated)
        del self.model  # Clean memory

    def recreate(self, X):
        self.model = self.od_method_constructor(**self.hyperparameters).fit(X)
        return self.model

# Defining a class for each type of model used in the evaluation process

class AutoIForest(OutlierDetectionMethod):
    od_method_constructor = IForest

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoIForest.od_method_constructor(**hyperparameters)

    def fit(self, X):
        self.model.fit(X)
        

class AutoLOF(OutlierDetectionMethod):
    od_method_constructor = LOF

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoLOF.od_method_constructor(**hyperparameters)

    def fit(self, X):
        self.model.fit(X)


class AutoOCSVM(OutlierDetectionMethod):
    od_method_constructor = OCSVM

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.model = AutoOCSVM.od_method_constructor(**hyperparameters)
        self.ocsvm_max_train = 10000

    def fit(self, X):
        if len(X) > self.ocsvm_max_train:
            self.model.fit(random.choices(np.array(X), k=self.ocsvm_max_train))
        else:
            self.model.fit(X)



          



