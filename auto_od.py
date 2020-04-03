from typing import Dict, Any

import hydro_serving_grpc as hs
import s3fs
from hydroserving.core.model.entities import UploadMetadata
from hydroserving.core.model.package import assemble_model
from hydroserving.core.model.service import ModelService
from hydroserving.core.monitoring.service import MonitoringService
from hydroserving.http.remote_connection import RemoteConnection
import pandas as pd

from utils import *

s3 = s3fs.S3FileSystem()


def deploy_monitoring_model(model_spec: Dict, modelspec_proto, model_path):
    """
    FIXME fill me
    :param modelspec_proto:
    :param model_spec:
    :param model_path:
    :return:
    """
    model_path = f'{model_path}/model'
    connection = RemoteConnection("http://localhost")  # FIXME move to arguments?
    monitoring_service = MonitoringService(connection)
    model_api = ModelService(connection, monitoring_service)
    tar_path = assemble_model(modelspec_proto, model_path)  # todo trouble with assemble
    metadata = UploadMetadata(
        name=model_spec['name'],
        host_selector=None,
        contract=model_spec['contract'],
        runtime=model_spec['runtime'],
        install_command=model_spec['install-command'],
        metadata={'status': '-'},  # todo
    )
    result = model_api.upload(tar_path, metadata)
    print(result)
    return 'Method _ for model _ is deploying'


def load_training_data(training_data_path, field_names):
    training_data = pd.read_csv(training_data_path)
    return training_data[field_names]


def pack_and_upload(monitoring_model, model_version_proto, auto_od_method):
    model_id = model_version_proto.id

    monitoring_model_path = create_folder_structure(auto_od_method.id, model_id)
    supported_tensors = TabularOD.get_compatible_tensors(model_version_proto.contract.predict.inputs)
    output_tensors = TabularOD.format_output_tensors(model_version_proto.contract.predict.outputs)

    monitoring_modelspec: Dict = generate_monitoring_modelspec(supported_tensors, output_tensors,
                                                               auto_od_method.id, model_id)

    # Saves monitoring model to folder specified in path alongside with requirements.txt and func.main
    # save_monitoring_objects(monitoring_model,
    # 						monitoring_modelspec,
    # 						auto_od_method.id,
    # 						auto_od_method.requirements,
    # 						path=monitoring_model_path)

    modelspec_proto = modelspec_dict_to_proto(monitoring_modelspec)

    deploy_monitoring_model(monitoring_modelspec,
                            modelspec_proto,
                            monitoring_model_path)


# FIXME change type annotations from Any to corresponding valid python type
def launch_auto_od(data_url: Any, model_version_proto: hs.manager.ModelVersion, method_id: Any, db: Any):
    """
    FIXME fill me
    :param data_url:
    :param model_version_proto:
    :param method_id:
    :param db:
    :return:
    """

    auto_od_method = TabularOD.from_id(method_id)

    # trackable_data = extract_data(model_proto, data_url)  # not now
    trackable_data = np.ones((10, 12))

    monitoring_model = auto_od_method.fit(trackable_data)
    pack_and_upload(monitoring_model, model_version_proto, auto_od_method)
    # delete_folder(model_id, method_id)  # FIXME asycn errors

    msg = "bla bla bla"
    status = 200
    return status, msg

# launch_auto_od("http://127.0.0.1:8000/adult.csv", 1, 1, 0)
