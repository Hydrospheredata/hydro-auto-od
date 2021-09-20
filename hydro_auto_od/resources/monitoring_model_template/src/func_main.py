import logging
import json
import joblib
from os import path
import pandas as pd

encoder = False

with open("/model/files/outlier_detector.joblib", "rb") as fp:
    od_model = joblib.load(fp)

with open("/model/files/fields_config.json", "r") as fp:
    config = json.load(fp)

if path.exists('/model/files/categorical_encoder.joblib'):

    with open("/model/files/categorical_encoder.joblib", "rb") as fp:
        categorical_encoder = joblib.load(fp)
    with open("/model/files/categorical_features.json", "r") as fp:
        cat_features = json.load(fp)
    cat_fields = cat_features["categorical_features"]
    encoder = True

FIELDS = config["field_names"]

def predict(**kwargs):

    x = pd.DataFrame.from_dict([kwargs])[FIELDS]
    if encoder:
        x[cat_fields] = categorical_encoder.transform(x[cat_fields])
    score = od_model.predict_proba(x, method='unify')[:,1]

    return {"value": score.item()}
