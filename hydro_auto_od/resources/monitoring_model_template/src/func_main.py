import os
import json
import joblib
import pandas as pd


with open("/model/files/outlier_detector.joblib", "rb") as fp:
    od_model = joblib.load(fp)
with open("/model/files/fields_config.json", "r") as fp:
    config = json.load(fp)

FIELDS = config['field_names']
encoder = False

if os.path.exists("/model/files/categorical_encoder.joblib"):
    with open("/model/files/categorical_encoder.joblib", "rb") as fp:
        categorical_encoder = joblib.load(fp)
    with open("/model/files/categorical_features.json", "r") as fp:
        cat_features = json.load(fp)
    CAT_FIELDS = cat_features['categorical_features']
    encoder = True

def predict(**kwargs):

    X = pd.DataFrame.from_dict([kwargs])[FIELDS]
    if encoder:
        X[CAT_FIELDS] = categorical_encoder.transform(X[CAT_FIELDS])
    score = od_model.predict_proba(X, method='unify')[:, 1]

    return {"value": score.item()}
