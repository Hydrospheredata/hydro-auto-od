import datetime
import glob
import json
import os
import sys
import tempfile
from enum import Enum
from multiprocessing import Process
from shutil import copytree
from time import sleep
from typing import List

import git
import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from hydro_serving_grpc.contract import ModelField, ModelContract
from hydrosdk.cluster import Cluster
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel, UploadResponse
from hydrosdk.monitoring import TresholdCmpOp, MetricSpecConfig, MetricSpec
from jsonschema import Draft7Validator
from pymongo import MongoClient
from pyod.models.hbos import HBOS
from s3fs import S3FileSystem
from waitress import serve
import sseclient
import requests
import json

from tabular_od_methods import TabularOD
from utils import get_monitoring_signature_from_monitored_signature, DTYPE_TO_NAMES

with open("version") as f:
    VERSION = f.read().strip()
    repo = git.Repo(".")
    BUILDINFO = {
        "version": VERSION,
        "gitHeadCommit": repo.active_branch.commit.hexsha,
        "gitCurrentBranch": repo.active_branch.name,
        "pythonVersion": sys.version
    }

with open("./schemas/auto_metric.json") as f:
    auto_metric_request_json_validator = Draft7Validator(json.load(f))


class AutoODMethodStatuses(Enum):
    PENDING = 0
    STARTED = 1
    SELECTING_MODEL = 2
    SELECTING_PARAMETERS = 3
    DEPLOYING = 4
    SUCCESS = 5
    FAILED = 6
    NOT_SUPPORTED = 7


def get_mongo_client():
    return MongoClient(host=MONGO_URL, port=MONGO_PORT,
                       username=MONGO_USER, password=MONGO_PASS,
                       authSource=MONGO_AUTH_DB)


HS_CLUSTER_ADDRESS = os.getenv("HS_CLUSTER_ADDRESS", "http://localhost")

MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
AUTO_OD_DB_NAME = os.getenv("AUTO_OD_DB_NAME", "auto_od")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
DEBUG_ENV = bool(os.getenv("DEBUG", True))

hs_cluster = Cluster(HS_CLUSTER_ADDRESS)
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
    Return build information - Python Version, build commit and service version
    :return:
    """
    return jsonify(BUILDINFO)


@app.route('/auto_metric', methods=['GET'])
def status():
    """
    # Checks status of training job in mongodb
    :return:
    """
    expected_args = {"monitored_model_version_id"}
    if set(request.args.keys()) != expected_args:
        return jsonify({"message": f"Expected args: {expected_args}. Provided args: {set(request.args.keys())}"}), 400

    model_version_id: int = int(request.args['monitored_model_version_id'])

    model_status = db.model_statuses.find_one({'monitored_model_version_id': model_version_id})

    if not model_status:
        return jsonify({"state": AutoODMethodStatuses.PENDING.name,
                        "description": "Training job for this model version was never requested."})
    else:
        return jsonify({"state": model_status['state'],
                        "description": model_status['description']})


@app.route('/auto_metric', methods=['POST'])
def launch():
    """
    Handles requests to create monitoring metrics for newly deployed model versions.
    Expected to be called from sonar.
    After checking that input parameters are valid, a new process will be created, and Response with HTTP code 202 will
    be returned from. Newly created process will update state in MongoDB as specified in Readme.MD and will train outlier detection model,
    deploy it, and attach to monitored model.
    :return:
    """

    job_config = request.get_json()

    if not auto_metric_request_json_validator.is_valid(job_config):
        error_message = "\n".join([error.message for error in auto_metric_request_json_validator.iter_errors(job_config)])
        return jsonify({"message": error_message}), 400

    training_data_path = job_config['training_data_path']

    monitored_model_version_id: int = job_config['monitored_model_version_id']

    code, message = process_auto_metric_request(training_data_path, monitored_model_version_id)
    return Response(status=code), jsonify({"message": message})


def process_auto_metric_request(training_data_path, monitored_model_version_id) -> (int, str):
    try:
        model_version = Model.find_by_id(hs_cluster, monitored_model_version_id)
    except ValueError:
        return 400, f"Model id={monitored_model_version_id} not found"

    if db.model_statuses.find_one({"monitored_model_version_id": monitored_model_version_id}):
        return 409, f"Training job already requested for model id={monitored_model_version_id}"

    if TabularOD.supports_signature(model_version.contract.predict):
        p = Process(target=train_and_deploy_monitoring_model,
                    args=(monitored_model_version_id, training_data_path))
        p.start()
        return 202, f"Started training job"
    else:
        model_status = db.model_statuses.insert_one({'monitored_model_version_id': monitored_model_version_id,
                                                     'training_data_path': training_data_path,
                                                     'state': AutoODMethodStatuses.NOT_AVAILABLE.name,
                                                     'description': "There are 0 supported fields in this model signature. "
                                                                    "To see how you can support AutoOD metric refer to the documentation"})

        return 200, f"Model state is {model_status['state']}, {model_status['description']}"


def train_and_deploy_monitoring_model(monitored_model_version_id, training_data_path):
    """
    This function:
    1. Downloads training data from S3 into pd.Dataframe
    2. Uses this training data to train HBOS outlier detection model
    3. Packs this model into temporary folder, and then into LocalModel
    4. Uploads this LocalModel to the cluster
    5. After this model finishes assemlby, attach it as a metric to the monitored model

    :param monitored_model_version_id:
    :param training_data_path: path pointing to s3
    :return:
    """
    # This method is intended to be used in another process,
    # so we need to create new MongoClient after fork
    mongo_client = get_mongo_client()
    db = mongo_client[AUTO_OD_DB_NAME]
    db.model_statuses.insert_one({'monitored_model_version_id': monitored_model_version_id,
                                  'training_data_path': training_data_path,
                                  'state': AutoODMethodStatuses.STARTED.name,
                                  'description': f"AutoOD training job started at {datetime.datetime.now()}"})

    monitored_model = Model.find_by_id(hs_cluster, monitored_model_version_id)

    supported_input_fields = TabularOD.get_compatible_fields(monitored_model.contract.predict.inputs)
    supported_output_fields = TabularOD.get_compatible_fields(monitored_model.contract.predict.outputs)

    # FIXME. right now there are no outputs in file provided? to reproduce use
    # {"monitored_model_version_id": 83,
    #  "training_data_path": "s3://feature-lake/training-data/83/training_data901590018946752001483.csv"
    #  }

    supported_fields: List[ModelField] = supported_input_fields  # + supported_output_fields
    supported_fields_names: List[str] = [field.name for field in supported_fields]
    supported_fields_dtypes: List[str] = [field.dtype for field in supported_fields]

    if S3_ENDPOINT:
        s3 = S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT})
        training_data = pd.read_csv(s3.open(training_data_path, mode='rb'))[supported_fields_names]
    else:
        training_data = pd.read_csv(training_data_path)[supported_fields_names]

    # Train HBOS on training data from S3
    outlier_detector = HBOS()
    outlier_detector.fit(training_data)

    db.model_statuses.update_one({'monitored_model_version_id': monitored_model_version_id},
                                 {"$set": {'state': AutoODMethodStatuses.DEPLOYING.name,
                                           'description': "Uploading metric to the cluster"}})

    try:
        # Create temporary directory to copy monitoring model payload there and delete folder later after uploading it to the cluster
        with tempfile.TemporaryDirectory() as tmpdirname:
            monitoring_model_folder_path = f"{tmpdirname}/{monitored_model.name}v{monitored_model.version}_auto_metric"
            copytree("monitoring_model_template", monitoring_model_folder_path)
            joblib.dump(outlier_detector, f'{monitoring_model_folder_path}/outlier_detector.joblib')

            # Save names and dtypes of analysed model fields to use in handling new requests in func_main.py
            monitoring_model_config = {"field_names": supported_fields_names,
                                       "field_dtypes": [DTYPE_TO_NAMES[x] for x in supported_fields_dtypes]}
            with open(f"{monitoring_model_folder_path}/fields_config.json", "w+") as fields_config_file:
                json.dump(monitoring_model_config, fields_config_file)

            payload_filenames = [os.path.basename(path) for path in glob.glob(f"{monitoring_model_folder_path}/*")]

            monitoring_model_contract = ModelContract(
                predict=get_monitoring_signature_from_monitored_signature(monitored_model.contract.predict))
            auto_od_metric_name = monitored_model.name + "_metric"

            local_model = LocalModel(name=auto_od_metric_name,
                                     contract=monitoring_model_contract,
                                     payload=payload_filenames,
                                     path=monitoring_model_folder_path,
                                     metadata={"created_by": "hydro_auto_od"},
                                     install_command="pip install -r requirements.txt",
                                     runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None))

            upload_response: UploadResponse = local_model._LocalModel__upload(hs_cluster)
            #while upload_response.building():
            #    pass
            response = requests.get(HS_CLUSTER_ADDRESS + "/api/v2/events", stream=True)
            client = sseclient.SSEClient(response)
            for event in client.events():
                print(event.data)
                if event.event == "ModelUpdate":
                    data = json.loads(event.data)
                    if data.get("id") == upload_response.model.id and data.get("status") != "Assembling":
                        break
                #pprint.pprint(json.loads(event.data))
    except Exception as e:
        print("Error!")
        print(e)
        db.model_statuses.update_one({'monitored_model_version_id': monitored_model_version_id},
                                     {"$set": {'state': AutoODMethodStatuses.FAILED.name,
                                               'description': f"Failed to pack & deploy monitoring model to a cluster - {str(e)}"}})
        return -1

    try:
        # Check that this model is found in the cluster
        monitoring_model = Model.find_by_id(hs_cluster, upload_response.model.id)

    except Exception as e:
        print("Error")
        print(e)
        db.model_statuses.update_one({'monitored_model_version_id': monitored_model_version_id},
                                     {"$set": {'state': AutoODMethodStatuses.FAILED.name,
                                               'description': f"Failed to find deployed monitoring model in a cluster - {str(e)}"}})

    sleep(10)  # TODO make proper waiting for the end of the monitoring model assembly
    try:
        # Add monitoring model to the monitored model
        metric_config = MetricSpecConfig(monitoring_model.id,
                                         outlier_detector.threshold_,
                                         TresholdCmpOp.LESS)
        MetricSpec.create(hs_cluster, "auto_od_metric", monitored_model.id, metric_config)
    except Exception as e:
        db.model_statuses.update_one({'monitored_model_version_id': monitored_model_version_id},
                                     {"$set": {'state': AutoODMethodStatuses.FAILED.name,
                                               'description': f"Failed to attach deployed monitoring model as a metric - {str(e)}"}})
        return -1

    db.model_statuses.update_one({'monitored_model_version_id': monitored_model_version_id},
                                 {"$set": {'state': AutoODMethodStatuses.SUCCESS.name, 'description': "😃"}})

    return 1


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
