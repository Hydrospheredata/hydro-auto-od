import time
from collections import namedtuple
from typing import Dict, Any

import numpy as np
import pyarrow.parquet as pq
import s3fs
from cli.hydroserving.core.model.package import assemble_model
from cli.hydroserving.core.model.service import ModelService
from cli.hydroserving.core.model.upload import upload_model_async
from cli.hydroserving.core.monitoring.service import MonitoringService
from cli.hydroserving.http.remote_connection import RemoteConnection
from tabular_od_methods import TabularOutlierDetectionMethod

from utils import generate_monitoring_modelspec, \
    gen_monitoring_script, save_monitoring_objects, create_folder_structure

s3 = s3fs.S3FileSystem()


def deploy_monitoring_model(model_spec, model_path):
    """
    FIXME fill me
    :param model_spec:
    :param model_path:
    :return:
    """
    connection = RemoteConnection("http://localhost")
    monitoring_service = MonitoringService(connection)
    model_api = ModelService(connection, monitoring_service)
    tar_path = assemble_model(model_spec, model_path)  # todo trouble with assemble

    upload_model_async(
        model_api=model_api,
        model=model_spec,
        model_path=tar_path)

    # todo delete the tar and folder
    return 'Method _ for model _ is deploying'


# FIXME change type annotations from Any to corresponding valid python type
def launch_auto_od(data_url: Any, model_proto: Any, method_id: Any, db: Any):
    """
    FIXME fill me
    :param data_url:
    :param model_proto:
    :param method_id:
    :param db:
    :return:
    """
    # todo check if method_id is in model status

    auto_od_method = TabularOutlierDetectionMethod.get_method(method_id)
    model_id = model_proto.id

    model_status = db.model_statuses.find_one({'model_id': model_id})
    supported_tensors = TabularOutlierDetectionMethod.get_compatible_tensors(model_proto.contract.predict.inputs)

    data = pq.ParquetDataset(data_url, filesystem=s3).read_pandas().to_pandas()
    trackable_data = data[[tensor['name'] for tensor in supported_tensors]]
    trackable_data = np.ones((10, 12))

    monitoring_model = auto_od_method.fit(trackable_data)

    # FIXME create_folder_structure -> something more verbose
    monitoring_model_path = create_folder_structure(method_id, model_id)

    # TODO Create object instead of dict
    monitoring_modelspec: Dict = generate_monitoring_modelspec(supported_tensors, model_proto.contract.predict.outputs, method_id, model_id)

    monitoring_pyscript = gen_monitoring_script(model_proto, method_id)

    save_monitoring_objects(monitoring_model, monitoring_pyscript, method_id, path=monitoring_model_path)  # \/
    monitoring_attributes_for_deploy = {'name': monitoring_modelspec['name'], 'payload': monitoring_modelspec['payload']}
    monitoring_attributes_for_deploy_object = namedtuple('Struct', monitoring_attributes_for_deploy.keys())(
        *monitoring_attributes_for_deploy.values())
    time.sleep(2)  # ?
    deploy_monitoring_model(monitoring_attributes_for_deploy_object, monitoring_model_path)

    status = 'successful'  # todo ENUMS instead of strings for statuses

    return status

# launch_auto_od("http://127.0.0.1:8000/adult.csv", 1, 0, 0)
# TODO
# need some stable data format to impement it
# trouble with assemble
# req.txt?
# connect od_model to model
# delete folder & zip
