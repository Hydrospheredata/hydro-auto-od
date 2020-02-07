import json
import os
import sys

import grpc
import hydro_serving_grpc as hs
import yaml
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from pymongo import MongoClient
from tabular_od_methods import TabularOutlierDetectionMethod, AutoHBOS
from waitress import serve

from auto_od import launch_auto_od

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))

with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version

app = Flask(__name__)
CORS(app)

MONGO_URL = os.getenv("MONGO_URL", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
AUTO_OD_COLLECTION_NAME = "auto_od"


def get_mongo_client():
    return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
                       username=MONGO_USER, password=MONGO_PASS,
                       authSource=MONGO_AUTH_DB)


client = get_mongo_client()
db = client['AUTO_OD_COLLECTION_NAME']


# FIXME change localhost to other URL SYsEnv
MANAGER_URL = os.getenv("MANAGER_URL", "manager:9090")
channel = grpc.insecure_channel("localhost:9090")
stub = hs.manager.ManagerServiceStub(channel)

TABULAR_METHODS = [AutoHBOS()]
MOCK_DATA_URL = 's3://hydroserving-dev-feature-lake/adult_classification/_hs_model_incremental_version=1/035a9a387fcc4240a91470d0843f0d9d' \
                '.parquet '


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am auto_ad Service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    return jsonify(BUILDINFO)


@app.route('/status/<int:model_id>', methods=['GET'])
def get_status(model_id):
    # d_type = 'tabular'
    model_request = hs.manager.GetVersionRequest(id=model_id)
    model_meta = stub.GetVersion(model_request)
    supported_tensors = TabularOutlierDetectionMethod.get_compatible_tensors(model_meta.contract.predict.inputs)

    if not supported_tensors:
        # TODO check error message
        response_json = {'auto_od_available': False,
                         'description': 'Automatic Outlier Detection is not available for this contract.'
                                        'check link:... for details'}
        return response_json

    model_status = db.model_statuses.find_one({'model_id': model_id})
    if not model_status:
        method_statuses = [{'name': method.name,
                            'id': method.id,
                            'status': "Not_launched".upper()} for method in TABULAR_METHODS]

        db.model_statuses.insert_one({'model_id': model_id, 'method_statuses': method_statuses})
    else:
        method_statuses = model_status['method_statuses']

        # TODO check description message
    # name = supported_tensors[0].name
    return jsonify({'auto_od_available': True,
                    'description': f'Trackable tensors: {[tensor["name"] for tensor in supported_tensors]}',
                    'methods': method_statuses})


@app.route('/launch/<int:model_id>/<int:method_id>', methods=['POST'])
def launch(model_id, method_id):
    data_url = MOCK_DATA_URL
    model_meta = stub.GetVersion(hs.manager.GetVersionRequest(id=model_id))
    status_code, msg = launch_auto_od(data_url, model_meta, method_id, db)
    return jsonify({"message": msg}), status_code


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
