import datetime
import glob
import json
import os
import sys
import tempfile
from enum import Enum
from shutil import copytree
from typing import List

import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from hydro_serving_grpc.contract import ModelField, ModelContract
from hydrosdk.cluster import Cluster
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel
from pymongo import MongoClient
from pyod.models.hbos import HBOS
from waitress import serve

from tabular_od_methods import TabularOD
from utils import get_monitoring_signature_from_monitored_signature

with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version


class AutoODMethodStatuses(Enum):
    PENDING = 0
    STARTED = 1
    SELECTING_MODEL = 2
    SELECTING_PARAMETERS = 3
    DEPLOYING = 4
    SUCCESS = 5
    FAILED = -1
    NOT_SUPPORTED = -2


def get_mongo_client():
    return MongoClient()
    # return MongoClient(host=MONGO_URL, port=MONGO_PORT,
    #                    username=MONGO_USER, password=MONGO_PASS,
    #                    authSource=MONGO_AUTH_DB)


HS_CLUSTER_ADDRESS = os.getenv("HS_CLUSTER_ADDRESS")

MONGO_URL = os.getenv("MONGO_URL")
MONGO_PORT = int(os.getenv("MONGO_PORT"))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
AUTO_OD_DB_NAME = os.getenv("AUTO_OD_DB_NAME", "auto_od")

DEBUG_ENV = bool(os.getenv("DEBUG", True))

hs_cluster = Cluster("https://hydro-serving.dev.hydrosphere.io")
mongo_client = get_mongo_client()
db = mongo_client[AUTO_OD_DB_NAME]

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am auto_ad service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    """

    :return:
    """
    return jsonify(BUILDINFO)


@app.route('/auto_metric', methods=['GET'])
def status():
    """
    # TODO write pydoc
    :return:
    """

    # TODO verify args provided to match OpenAPI
    expected_args = {"model_version_id"}

    model_version_id: int = int(request.args['model_version_id'])

    model_status = db.model_statuses.find_one({'model_id': model_version_id})

    if not model_status:
        return jsonify({"state": AutoODMethodStatuses.PENDING, "description": "Training job for this model version was never requested."})
    else:
        return jsonify({"state": model_status['state'], "description": model_status['description']})


@app.route('/auto_metric', methods=['POST'])
def launch():
    """
    # TODO write pydoc
    :return:
    """
    # TODO add jsonschema to this route to validate input

    job_config = request.get_json()
    training_data_path = job_config['training_data_path']

    model_version_id: int = job_config['model_version_id']

    try:
        model_version = Model.find_by_id(hs_cluster, model_version_id)
    except ValueError:
        return Response(400), f"Error, unable to find Model by id={model_version_id} in cluster={hs_cluster}"

    if db.model_statuses.find_one({"model_version_id": model_version_id}):
        return Response(409), f"Auto OD Training job was already launched for this model version"

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

    # FIXME. right now there are no outputs in file provided? to reproduce use
    # {"model_version_id": 83,
    #  "training_data_path": "s3://feature-lake/training-data/83/training_data901590018946752001483.csv"
    #  }
    supported_fields: List[ModelField] = supported_input_fields  # + supported_output_fields
    supported_fields_names: List[str] = [field.name for field in supported_fields]

    training_data = pd.read_csv(training_data_path)[supported_fields_names]

    # Train HBOS on training data from S3
    outlier_detector = HBOS()
    outlier_detector.fit(training_data)

    # Create temporary directory to copy monitoring model payload there and delete folder later after uploading it to the cluster
    with tempfile.TemporaryDirectory() as tmpdirname:
        monitoring_model_folder_path = f"{tmpdirname}/{model_version.name}v{model_version.version}_auto_metric"
        copytree("monitoring_model_template", monitoring_model_folder_path)
        joblib.dump(outlier_detector, f'{monitoring_model_folder_path}/outlier_detector.joblib')

        db.model_statuses.update_one({'model_version_id': model_version_id}, {"$set": {'state': AutoODMethodStatuses.DEPLOYING.name,
                                                                                       'description': "Uploading metric to the cluster"}})

        payload_filenames = [os.path.basename(path) for path in glob.glob(f"{monitoring_model_folder_path}/*")]

        monitoring_model_contract = ModelContract(predict=get_monitoring_signature_from_monitored_signature(model_version.contract.predict))
        auto_od_metric_name = model_version.name + "_metric"

        # TODO discuss which metadata should be provided here
        local_model = LocalModel(name=auto_od_metric_name,
                                 contract=monitoring_model_contract,
                                 payload=payload_filenames,
                                 path=monitoring_model_folder_path,
                                 runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None))

        # TODO how to check that it succeded? TODO change status based on upload response!
        upload_response = local_model._LocalModel__upload(hs_cluster)

    return "ok"


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
