import json
import joblib
import numpy as np

with open("/model/files/outlier_detector.joblib", "rb") as fp:
    od_model = joblib.load(fp)

with open("/model/files/fields_config.json", "r") as fp:
    config = json.load(fp)

FIELDS = config['field_names']

def predict(**kwargs):
    x = np.array([kwargs.get(field_name) for field_name in FIELDS], dtype=float)
    score = od_model.predict_proba(x.reshape(1, -1), method='unify')[:, 1]
    return {"value": score.item()}
