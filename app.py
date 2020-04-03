import datetime
import glob
import json
import logging
import os
import pathlib
import sys
from enum import Enum
from typing import List

import joblib
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from hydro_serving_grpc.contract import ModelField, ModelSignature, ModelContract
from hydrosdk.cluster import Cluster
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel
from pymongo import MongoClient
from pyod.models.hbos import HBOS
from waitress import serve

import auto_od
from tabular_od_methods import TabularOD, AutoHBOS

with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version


class AutoODMethodStatuses(Enum):
    PENDING = 0
    STARTED = 1
    SELECTING_MODEL = 2
    DEPLOYING = 5
    SUCCESS = 6
    FAILED = -1
    NOT_AVAILABLE = -2


def get_mongo_client():
    return MongoClient()
    # return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
    #                    username=MONGO_USER, password=MONGO_PASS,
    #                    authSource=MONGO_AUTH_DB)


app = Flask(__name__)
CORS(app)

MONGO_URL = os.getenv("MONGO_URL", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")

HS_CLUSTER_ADDRESS = os.getenv("HS_CLUSTER_ADDRESS")

AUTO_OD_COLLECTION_NAME = "auto_od"

# hs_cluster = Cluster(HS_CLUSTER_ADDRESS)

hs_cluster = Cluster("https://hydro-serving.dev.hydrosphere.io")

TABULAR_METHODS = [AutoHBOS()]

MOCK_TRAINING_DATA_PATH = "s3://feature-lake/training-data/10/training_data175426489644208838110.csv"

mongo_client = get_mongo_client()
db = mongo_client[AUTO_OD_COLLECTION_NAME]


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am auto_ad service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    return jsonify(BUILDINFO)


@app.route('/auto_od_metric', methods=['GET'])
def status():
    expected_args = {"model_version_id"}

    model_version_id: int = int(request.args['model_version_id'])

    model_status = db.model_statuses.find_one({'model_id': model_version_id})

    if not model_status:
        # FIXME check for model version in cluster and return appropriate HTTP Code (which allows retrying the request)
        return Response(400)
    else:
        del model_status['_id']
        return jsonify(model_status)


@app.route('/auto_od_metric', methods=['POST'])
def launch():
    # TODO add jsonschema to this route

    job_config = request.get_json()
    training_data_path = job_config['training_data_path']

    model_version_id: int = job_config['model_version_id']
    model_version = Model.find_by_id(hs_cluster, model_version_id)

    logging.info(training_data_path)
    # TODO check for support, if not supported - update status and return it

    if TabularOD.supports_signature(model_version.contract.predict):
        model_status = db.model_statuses.insert_one({'model_version_id': model_version_id,
                                                     'training_data_path': training_data_path,
                                                     'state': AutoODMethodStatuses.STARTED.name,
                                                     'description': f"AutoOD training job started at {datetime.datetime.now()}"})
    else:
        model_status = db.model_statuses.insert_one({'model_version_id': model_version_id,
                                                     'training_data_path': training_data_path,
                                                     'state': AutoODMethodStatuses.NOT_AVAILABLE.name,
                                                     'description': "There are 0 supported fields in this model signature. "
                                                                    "To see how you can support AutoOD metric refer to the documentation"})

        return jsonify({"state": model_status['state'], "description": model_status['description']})

    supported_input_fields = TabularOD.get_compatible_fields(model_version.contract.predict.inputs)
    supported_output_fields = TabularOD.get_compatible_fields(model_version.contract.predict.outputs)
    supported_fields: List[ModelField] = supported_input_fields  # + supported_output_fields # FIXME right now there are no outputs in file?
    supported_fields_names: List[str] = [field.name for field in supported_fields]

    training_data = auto_od.load_training_data(training_data_path, supported_fields_names)

    outlier_detector = HBOS()
    outlier_detector.fit(training_data)

    filename = 'monitoring_model_template/outlier_detector.joblib'
    joblib.dump(outlier_detector, filename)

    db.model_statuses.update_one({'model_version_id': model_version_id}, {"$set": {'state': AutoODMethodStatuses.DEPLOYING.name,
                                                                                   'description': "Uploading trained metric to the cluster"}})

    payload = glob.glob("monitoring_model_template/*")
    path = pathlib.Path("monitoring_model_template").absolute()

    monitored_model_signature = model_version.contract.predict.signature
    monitoring_model_signature = ModelSignature(inputs=monitored_model_signature.inputs + monitored_model_signature.outputs,
                                                outputs=[ModelField(name="metric_value", shape=[])])
    monitoring_model_contract = ModelContract(predict=monitoring_model_signature)
    auto_od_metric_name = model_version.name + "_metric"
    local_model = LocalModel(name=auto_od_metric_name,
                             contract=monitoring_model_contract,
                             payload=payload,
                             path=path,
                             runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None))

    upload_response = local_model._LocalModel__upload(hs_cluster)

    return 200


# def get_contract(signature):


if __name__ == "__main__":
    DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
