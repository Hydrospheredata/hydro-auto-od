# This realization is based in two projects combined to
# find an optimal model together with a hyperparameter.
# Specifically Mass Volume is used to maintain the process
# of Hydro Automatic Outlier Detection.

# Model Selection
# https://github.com/ngoix/EMMV_benchmarks

# Hyperparameter selection
# https://github.com/albertcthomas/anomaly_tuning

# NOTES

# 1) Working only with PyOD realization of algorithms
# 2) Currently models are presented by IForest, LOF and OCSVM
# 3) Hyperparameter is being searched for LOF and IForest only
# 4) Contamination parameter is assigned to 4%. 


import numpy as np
import logging 
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.model_selection import train_test_split
from hydro_auto_od.tuning import model_tuning, high_tuning, low_tuning


models = {'IForest': IForest, 'LOF': LOF, 'OCSVM': OCSVM}
algo_param = {
    'LOF': {'n_neighbors': np.arange(5,31)},
    'IForest': {'n_estimators': np.array([20, 50, 100, 150, 200, 250])},
}

def model_selection(data: pd.DataFrame):

    X = np.array(data)
    
    x_train, x_test = train_test_split(X, test_size = 0.2)

    # Evaluating each model among candidates
    if X.shape[1] <= 7:
        chosen_model = low_tuning(x_train, x_test, list(models.values()), base_estimator=None,
                                  alphas=np.arange(0.9, 0.99, 0.001))
    else:
        chosen_model = high_tuning(x_train, x_test, list(models.values()), base_estimator=None,
                                   alphas=np.arange(0.9, 0.99, 0.001), averaging=50)
      
    chosen_name = chosen_model.__name__
    if chosen_name == 'OCSVM':
        final_model = chosen_model(**{'contamination': 0.03})
    else:
        # Choosing hyperparameter
        parameters = algo_param[chosen_name]
        chosen_params = model_tuning(x_train, x_test, base_estimator=chosen_model,
                                     parameters=parameters, alphas=np.arange(0.05, 1., 0.05))
        chosen_params['contamination'] = 0.03
        final_model= chosen_model(**chosen_params)

    final_model.fit(X)
    return final_model
