
from auto_od_methods import AutoIForest, AutoLOF, AutoOCSVM
from sklearn.model_selection import train_test_split
import numpy as np
import logging




iforest_hyperparams = [{"n_estimators":100}, {"n_estimators":20}, {"n_estimators":50}, 
                      {"n_estimators":150}, {"n_estimators":200}, {"n_estimators":250}]

lof_hyperparams = [{"n_neighbors":20}, {"n_neighbors":5}, {"n_neighbors":6}, {"n_neighbors":7}, 
                  {"n_neighbors":8}, {"n_neighbors":15}, {"n_neighbors":25}]
lof_hyperparams = [dict(kwargs, **{'novelty': True}) for kwargs in lof_hyperparams]

ocsvm_hyperparams = [{"kernel":'rbf'}, {'kernel':'poly'}]


def model_selection(X):

  x_train, x_test = train_test_split(X, test_size = 0.3)

  iforest_models = [AutoIForest(kwargs) for kwargs in iforest_hyperparams]

  lof_models = [AutoLOF(kwargs) for kwargs in lof_hyperparams]
  ocsvm_models = [AutoOCSVM(kwargs) for kwargs in ocsvm_hyperparams]


  for model in iforest_models + lof_models + ocsvm_models:
    model.fit(x_train)
    model.evaluate(X, x_test)

  best_mv_model = min(iforest_models + lof_models + ocsvm_models, key=lambda x: x.mv)
  best_em_model = max(iforest_models + lof_models + ocsvm_models, key=lambda x: x.em)

  if best_mv_model.mv < 0:
    logging.info('MV value is lower than zero. Applying best model by EM.')
    outlier_detector = min(iforest_models + lof_models + ocsvm_models, key=lambda x: x.em)
  else:
    outlier_detector = best_mv_model

    if best_em_model == best_mv_model:
      logging.info('EM-MV convergence is successful!')
    else:
      logging.info('Applying best model by MV.')

  return outlier_detector
