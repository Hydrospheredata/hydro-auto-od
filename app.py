import json
import os
import sys
from enum import Enum
import logging

import grpc
import hydro_serving_grpc as hs
import pymongo
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

logger = logging.getLogger()


# TODO add logging!


class AutoODMethodStatuses(Enum):
	NOT_LAUNCHED = 1
	LAUNCHED = 2
	TRAINING = 3
	EVALUATING = 4
	DEPLOYING = 5
	DEPLOYED = 6
	NOT_SUPPORTED = 7
	FAILED = 9


def get_mongo_client():
	return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
					   username=MONGO_USER, password=MONGO_PASS,
					   authSource=MONGO_AUTH_DB)


TABULAR_METHODS = [AutoHBOS()]

# client = get_mongo_client()   #fixme
# ---
client = MongoClient('localhost', 27017)
db = client['AUTO_OD_COLLECTION_NAME']
db.model_statuses.insert_one({'model_id': 1,
							  'methods_statuses': dict([(method.name, {'name': method.name,
																	   'status': AutoODMethodStatuses.NOT_LAUNCHED.value})
														for method in
														TABULAR_METHODS]),
							  "description": 'description'
							  })
# ---
# FIXME change localhost to other URL SYsEnv
MANAGER_URL = os.getenv("MANAGER_URL", "manager:9090")
channel = grpc.insecure_channel("localhost:9090")
stub = hs.manager.ManagerServiceStub(channel)

MOCK_DATA_URL = 's3://hydroserving-dev-feature-lake/adult_classification/_hs_model_incremental_version=1/035a9a387fcc4240a91470d0843f0d9d' \
				'.parquet '


def hello():
	return "Hi! I am auto_ad Service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
	return jsonify(BUILDINFO)


@app.route('/status/<int:model_id>', methods=['GET'])
def status(model_id):
	model_status = db.model_statuses.find_one({'model_id': model_id})
	
	if not model_status:
		
		model_request = hs.manager.GetVersionRequest(id=model_id)
		model_meta = stub.GetVersion(model_request)
		supported_tensors = TabularOutlierDetectionMethod.get_compatible_tensors(model_meta.contract.predict.inputs)
		
		if not supported_tensors:
			methods_statuses = dict([(method.name, {'name': method.name,
													'status': AutoODMethodStatuses.NOT_SUPPORTED.value}) for method in
									 TABULAR_METHODS])
		else:
			# Add initial statuses for all supported methods if this model has never interacted with auto_od service
			methods_statuses = dict([(method.name, {'name': method.name,
													'status': AutoODMethodStatuses.NOT_LAUNCHED.value}) for method in
									 TABULAR_METHODS])
		
		description = f"Only float and int scalars are analysed. Analysed tensors: [{supported_tensors}]"
		db.model_statuses.insert_one({'model_id': model_id,
									  'methods_statuses': methods_statuses,
									  "description": description
									  })
	else:
		methods_statuses, description = model_status['methods_statuses'], model_status['description']
	
	return jsonify(
		{'auto_od_available': any(
			[m['status'] != AutoODMethodStatuses.NOT_SUPPORTED.value for method_id, m in methods_statuses.items()]),
			'description': description,
			'methods': methods_statuses})


@app.route('/launch/<int:model_id>/<int:method_id>', methods=['POST'])
def launch(model_id, method_id):
	# TODO validate for possible method_ids and model_ids
	
	training_data_url = MOCK_DATA_URL
	model_version_proto = stub.GetVersion(hs.manager.GetVersionRequest(id=model_id))
	
	# todo check for NOT_LAUNCHED status of model_id & method_id
	
	# TODO COSTIL
	method_name = TabularOutlierDetectionMethod.from_id(method_id).name
	
	model_status = db.model_statuses.find_one({'model_id': model_id}, sort=[('_id', pymongo.DESCENDING)])
	print(model_status)
	if model_status:
		if model_status['methods_statuses'][method_name]['status'] == AutoODMethodStatuses.NOT_LAUNCHED.value:
			
			db.model_statuses.find_one_and_update({'model_id': model_id},
												  {"$set": {
													  f"methods.{method_name}.status": AutoODMethodStatuses.LAUNCHED.value}})
			
			status_code, msg = launch_auto_od(training_data_url, model_version_proto, method_id, db)
			print(status_code, msg)  # FIXME change to logger.log("....")
			
			return jsonify({"message": msg}), status_code
		
		else:
			status_code, msg = 400, "keke"
			# RETURN INVALID STATE error
			return jsonify({"message": msg}), status_code
	
	else:
		status(model_id)
		return launch(model_id, method_id)


if __name__ == "__main__":
	if not DEBUG_ENV:
		serve(app, host='0.0.0.0', port=5000)
	else:
		app.run(debug=True, host='0.0.0.0', port=5000)
