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


def load_data(url):
    f = BytesIO(urlopen(url).read())
    data = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=object)
    return data


def choose_features(url):
    data = load_data(url)
    data = data[:, [0, 2, 4, 10, 11, 12]].astype(float)
    return data


def generate_monitoring_modelspec(inputs, outputs, method_id, model_id):
    name = TabularOutlierDetectionMethod.get_method(method_id).name

    method_meta = {"kind": "Model",
                   "name": f"{model_id}_{name}",
                   "payload": ["src/", "../requirements.txt", f"../{name}.model"],     #todo file with req!!
                   "runtime": "hydrosphere/serving-runtime-python-3.6:0.1.2-rc0",
                   "install-command": "pip install -r requirements.txt",
                   "contract": {
                       "name": "predict",
                       "inputs": inputs.extend(outputs),
                       "outputs": {"value": {"shape": 'scalar', "type": "double"}}}
                   }
    return method_meta




def gen_monitoring_script(model_meta, method_id):
    # TODO generate py script of monitoring model
    return 'filename'
    # os.touch()

def create_folder_structure(method_id, model_id):
    model_path = f'monitoring_model{model_id}-{method_id}'
    os.mkdir(model_path)
    os.mkdir(f'{model_path}/src')
    return model_path

def save_monitoring_objects(od_model, od_script, method_id, path):
    name = TabularOutlierDetectionMethod.get_method(method_id).name
    model_pickle_name = f'{path}/{name}.model'
    pickle.dump(od_model, open(model_pickle_name, 'wb'))
    with open(f'{path}/src/main.py', 'w') as script:
        script.write(od_script)

    #todo save .py file to path/src