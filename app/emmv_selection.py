
from auto_od_methods import AutoIForest, AutoLOF, AutoOCSVM
from sklearn.model_selection import train_test_split
import numpy as np


# Models with predefined hyperparameters

iforest_hyperparams = [{"n_estimators":100}, {"n_estimators":20}, {"n_estimators":50}, 
                      {"n_estimators":150}, {"n_estimators":200}, {"n_estimators":250}]
iforest_hyperparams = [dict(kwargs, **{'contamination': 0.04}) for kwargs in iforest_hyperparams]

lof_hyperparams = [{"n_neighbors":20}, {"n_neighbors":5}, {"n_neighbors":6}, {"n_neighbors":7}, 
                  {"n_neighbors":8}, {"n_neighbors":15}, {"n_neighbors":25}]
lof_hyperparams = [dict(kwargs, **{'contamination': 0.04}) for kwargs in lof_hyperparams]

ocsvm_hyperparams = [{"kernel":'rbf', 'nu':0.01, 'contamination':0.04}, {'kernel':'poly', 'nu':0.01, 'contamination':0.04}]


def model_selection(X):

  x_train, x_test = train_test_split(X, test_size = 0.3)

  iforest_models = [AutoIForest(kwargs) for kwargs in iforest_hyperparams]
  lof_models = [AutoLOF(kwargs) for kwargs in lof_hyperparams]
  ocsvm_models = [AutoOCSVM(kwargs) for kwargs in ocsvm_hyperparams]

  '''Dimension restriction for model's choice to speed up 
  	the process for high-dimensional cases
  '''

  if X.shape[1] > 170:
      candidates = iforest_models + ocsvm_models
  else:
      candidates = iforest_models + lof_models + ocsvm_models

  # Evaluating each model among candidates

  for model in candidates:
    model.fit(x_train)
    model.evaluate(X, x_test)

# Choose the model by the min MV score

  outlier_detector = min(candidates, key=lambda x: x.mv)


  return outlier_detector
