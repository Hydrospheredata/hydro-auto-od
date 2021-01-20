# This realization is based in two projects combined to
# find an optimal model together with a hyperparameter
# Specifically Mass Volume is used to maintain the process
# Hydro Automatic Outlier Detection

# Model Selection
# https://github.com/ngoix/EMMV_benchmarks

# Hyperparameter selection
# https://github.com/albertcthomas/anomaly_tuning

# NOTES

# 1) Working only with PyOD realization of algorithms
# 2) Currently candidates are presented by IForest, LOF and OCSVM
# 3) Hyperparameter is being searched for LOF and IForest only
# 4) Contamination parameter is assigned to 4%. Except of anomalies
# this helps to identify potential drifts in features or in overall
# data given to a nicely chosen model


import numpy as np
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from tuning import model_tuning, high_tuning, low_tuning
from sklearn.model_selection import train_test_split, ShuffleSplit


models = {'IForest': IForest, 'LOF': LOF, 'OCSVM': OCSVM}

algo_param = {
    'LOF': {'n_neighbors': np.arange(5,31)},
    'IForest': {'n_estimators': np.array([20, 50, 100, 150, 200, 250])}}

def model_selection(X):

    X = np.array(X)
    x_train, x_test = train_test_split(X, test_size = 0.2)

  # Evaluating each model among candidates
    if X.shape[1] <= 5:
        chosen_model = low_tuning(x_train, x_test, list(models.values()), base_estimator = None,
                                alphas = np.arange(0.9, 0.99, 0.001))
    else:
        chosen_model = high_tuning(x_train, x_test, list(models.values()), base_estimator = None,
                                   alphas = np.arange(0.9, 0.99, 0.001), averaging = 50)
      
    chosen_class = chosen_model.__class__
    chosen_name = chosen_model.__name__
    
    if chosen_name == 'OCSVM':
        final_model = chosen_model(**{'contamination': 0.04})
    else:

        # Choosing hyperparameter
        parameters = algo_param[chosen_name]
        param_name = list(parameters.keys())[0]
        cv = ShuffleSplit(n_splits=10, test_size=0.2)
        chosen_params = model_tuning(X = X, base_estimator = chosen_model,
                                parameters = parameters, cv = cv)
        counts_values = np.unique(list(i[param_name] for i in chosen_params), return_counts = True)
        hyperparam = counts_values[0][np.argmax(counts_values[1])]
        fin_param = {param_name: hyperparam, 'contamination': 0.04}
        final_model = chosen_model(**fin_param)

    final_model.fit(X)
    
    return final_model