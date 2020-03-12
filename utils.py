from io import BytesIO
import pickle
import numpy as np
from urllib.request import urlretrieve, urlopen
import urllib
import yml2json
import yaml, json
import os
from collections import namedtuple
from tabular_od_methods import TabularOutlierDetectionMethod
import shutil


def load_data(url):
	f = BytesIO(urlopen(url).read())
	data = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=object)
	return data


def choose_features(url):
	data = load_data(url)
	data = data[:, [0, 2, 4, 10, 11, 12]].astype(float)
	return data


def generate_monitoring_modelspec(inputs, outputs, method_id, model_id):
	name = TabularOutlierDetectionMethod.from_id(method_id).name
	inputs.extend(outputs)
	method_meta = {"kind": "Model",
				   "name": f"{model_id}_{name}",
				   "payload": ["src/", "../requirements.txt", f"../{name}.model"],  # todo file with req!!
				   "runtime": {
					   'name': 'hydrosphere/serving-runtime-python-3.6',
					   'tag': 'dev'},
				   "install-command": "pip install -r requirements.txt",
				   "contract": {
					   "modelName": f"{name}",
					   "predict": {
						   "signatureName": "predict",
						   "inputs": inputs,
						   "outputs":
							   [{'name': 'score',
								'dtype': 'DT_DOUBLE',
								'profile': 'NUMERICAL',
								 'shape': {'dim':[], 'unknownRank':False}}]
					   }
				   }
				   }
	return method_meta


def gen_monitoring_script(target):
	# TODO generate py script of monitoring model// copy file to tar

	return 'filename'


# os.touch()


def create_folder_structure(method_id, model_id):
	model_path = f'monitoring_model{model_id}-{method_id}'
	os.mkdir(model_path)
	os.mkdir(f'{model_path}/model')
	os.mkdir(f'{model_path}/model/src')
	return f'{model_path}'


def save_monitoring_objects(od_model, model_spec, method_id, requirements, path):
	name = TabularOutlierDetectionMethod.from_id(method_id).name
	model_pickle_name = f'{path}/{name}.model'
	pickle.dump(od_model, open(model_pickle_name, 'wb'))
	shutil.copy('func_main.py', f'{path}/model/src/func_main.py')
	with open(f'{path}/requirements.txt', 'w') as req_file:
		req_file.write(requirements)
	with open(f'{path}/model/serving.yaml', 'w') as yml_file:
		yaml.dump(model_spec, yml_file)


def delete_folder(model_id, method_id):
	shutil.rmtree(f'monitoring_model{model_id}-{method_id}')
	pass

def modelspec_dict_to_proto(monitoring_modelspec):
	monitoring_attributes_for_deploy = {'name': monitoring_modelspec['name'],
										'payload': monitoring_modelspec['payload']}
	monitoring_attributes_for_deploy_object = namedtuple('Struct',
														 monitoring_attributes_for_deploy.keys())(
		*monitoring_attributes_for_deploy.values())
	return monitoring_attributes_for_deploy_object


