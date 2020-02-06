
from flask import Flask, request, jsonify, url_for
from auto_od import launch_auto_od
from pymongo import MongoClient

import hydro_serving_grpc as hs
import grpc
import yaml
import json
from tabular_od_methods import TabularOutlierDetectionMethod, AutoHBOS

app = Flask(__name__)
client = MongoClient('localhost', 27017)
db = client.auto_od     #??

channel = grpc.insecure_channel("localhost:9090")
stub = hs.manager.ManagerServiceStub(channel)

TABULAR_METHODS = [AutoHBOS()]
# AutoHBOS.name

@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am auto_ad Service"


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
        
        db.model_statuses.insert_one({'model_id': model_id, 'method_statuses':  method_statuses})
    else:
        method_statuses = model_status['method_statuses'] 
    
    # TODO check description message
    # name = supported_tensors[0].name
    return jsonify({'auto_od_available': True,
                    'description': f'Trackable tensors: {[tensor["name"] for tensor in supported_tensors]}',
                    'methods': method_statuses})

@app.route('/launch/<int:model_id>/<int:method_id>/<str:data_url>', methods=['GET', 'POST'])
def launch(model_id, method_id, data_url):
    data_url = 's3://hydroserving-dev-feature-lake/adult_classification/_hs_model_incremental_version=1/035a9a387fcc4240a91470d0843f0d9d.parquet'
    
    model_meta = stub.GetVersion(hs.manager.GetVersionRequest(id=model_id))
    status_code, msg = launch_auto_od(data_url, model_meta, method_id, db)
    return jsonify({"message": msg}), status_code

# db.model_statuses.delete_one({'model_id': 1})
# get_status(1)
# launch(1,1)


'''
$ export FLASK_APP=app.py
$ flask run
'''