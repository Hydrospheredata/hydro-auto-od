import json

import hydro_serving_grpc as hs
import joblib
import numpy as np

with open("/model/files/outlier_detector.joblib", "rb") as fp:
    od_model = joblib.load(fp)

with open("/model/files/fields_config.json", "r") as fp:
    config = json.load(fp)

FIELDS = config['field_names']
FIELDS_DTYPES = config['field_dtypes']


def predict(**kwargs):
    fields_values = [getattr(kwargs[field_name], f"{field_dtype}_val") for field_name, field_dtype in zip(FIELDS, FIELDS_DTYPES)]

    x = np.array(fields_values, dtype=float).reshape(1, -1)

    score = od_model.predict_proba(x, method='unify')[:,1]

    metric_value_proto = hs.TensorProto(
        double_val=score.flatten().tolist(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto())

    return hs.PredictResponse(outputs={"value": metric_value_proto})
