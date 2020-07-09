
from auto_od_methods import AutoIForest, AutoLOF, AutoOCSVM
import numpy as np
import logging




iforest_hyperparams = [{"n_estimators":20}, {"n_estimators":50}, {"n_estimators":100}, 
                      {"n_estimators":150}, {"n_estimators":200}, {"n_estimators":250}]

lof_hyperparams = [{"n_neighbors":5}, {"n_neighbors":6}, {"n_neighbors":7}, {"n_neighbors":8}, 
                  {"n_neighbors":10}, {"n_neighbors":15}, {"n_neighbors":20}]

ocsvm_hyperparams = [{"kernel":'rbf'}, {'kernel':'poly'}]


def model_selection(X):
  n_samples, _ = np.shape(X)
  n_samples_train = n_samples // 2
  x_train = X[:n_samples_train, :]
  x_test = X[n_samples_train:, :]



  iforest_models = [AutoIForest(kwargs) for kwargs in iforest_hyperparams]
  lof_hyperparams = [dict(kwargs, **{'novelty': True}) for kwargs in lof_hyperparams]

  lof_models = [AutoLOF(kwargs) for kwargs in lof_hyperparams]
  ocsvm_models = [AutoOCSVM(kwargs) for kwargs in ocsvm_hyperparams]


  for model in iforest_models + lof_models + ocsvm_models:
    model.fit(x_train)
    model.evaluate(x_test)


  best_iforest = max(iforest_models)
  best_lof = max(lof_models)
  best_ocsvm = max(ocsvm_models)


  best_models = [best_iforest, best_lof, best_ocsvm]
  best_mv_model = min(best_models, key=lambda x: x.mv)
  best_em_model = max(best_models, key=lambda x: x.em)

  if best_em_model == best_mv_model:
    logging.info('EM-MV convergence is successful!')
    outlier_detector = best_mv_model
  else:
    logging.info('Applying best model by MV')
    outlier_detector = best_mv_model

  return outlier_detector
