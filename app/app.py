import datetime
import glob
import json
import logging
import os
import tempfile
from multiprocessing import Process
from shutil import copytree
from typing import List

import joblib
import pandas as pd
from hydro_serving_grpc.contract import ModelField, ModelContract
from hydrosdk.cluster import Cluster
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import ModelVersion, LocalModel, UploadResponse
from hydrosdk.monitoring import TresholdCmpOp, MetricSpecConfig, MetricSpec
# from pyod.models.hbos import HBOS
from emmv_selection import model_selection
from s3fs import S3FileSystem
from tabular_od_methods import TabularOD
from training_status_storage import TrainingStatusStorage, AutoODMethodStatuses, TrainingStatus
from utils import get_monitoring_signature_from_monitored_signature, DTYPE_TO_NAMES


S3_ENDPOINT = os.getenv("S3_ENDPOINT")
DEBUG_ENV = bool(os.getenv("DEBUG", True))

HS_CLUSTER_ADDRESS = os.getenv("HS_CLUSTER_ADDRESS", "http://localhost")
hs_cluster = Cluster(HS_CLUSTER_ADDRESS)


def process_auto_metric_request(training_data_path, monitored_model_version_id) -> (int, str):
    logging.info("Started processing auto-od request for model (%d)", monitored_model_version_id)

    try:
        model_version = ModelVersion.find_by_id(hs_cluster, monitored_model_version_id)
    except ModelVersion.NotFound:
        logging.error("Monitored model (%d) not found", monitored_model_version_id)
        return 400, f"Model id={monitored_model_version_id} not found"

    if TrainingStatusStorage.find_by_model_version_id(monitored_model_version_id) is not None:
        logging.info("%s: Training job already requested", repr(model_version))
        return 409, f"Training job already requested for model id={monitored_model_version_id}"

    if TabularOD.supports_signature(model_version.contract.predict):
        p = Process(target=train_and_deploy_monitoring_model,
                    args=(monitored_model_version_id, training_data_path))
        p.start()
        logging.info("%s: Created training job for model", repr(model_version))
        return 202, f"Started training job"
    else:
        logging.info("%s: signature is not supported", monitored_model_version_id)
        desc = ("There are 0 supported fields in this model signature. "
                "To see how you can support AutoOD metric refer to the documentation")
        model_status = \
            TrainingStatus(monitored_model_version_id, training_data_path, AutoODMethodStatuses.NOT_SUPPORTED, desc)
        TrainingStatusStorage.save_status(model_status)
        return 200, f"Model state is {model_status.state}, {model_status.description}"


def train_and_deploy_monitoring_model(monitored_model_version_id, training_data_path):
    """
    This function:
    1. Downloads training data from S3 into pd.Dataframe
    2. Uses this training data to train HBOS outlier detection model
    3. Packs this model into temporary folder, and then into LocalModel
    4. Uploads this LocalModel to the cluster
    5. After this model finishes assembly, attach it as a metric to the monitored model
    :param monitored_model_version_id:
    :param training_data_path: path pointing to s3
    :return:
    """
    # This method is intended to be used in another process,
    # so we need to create new MongoClient after fork
    description = f"AutoOD training job started at {datetime.datetime.now()}"
    model_status = \
        TrainingStatus(monitored_model_version_id, training_data_path, AutoODMethodStatuses.STARTED, description)
    TrainingStatusStorage.save_status(model_status)

    logging.info("Getting monitored model (%d)", monitored_model_version_id)
    monitored_model = ModelVersion.find_by_id(hs_cluster, monitored_model_version_id)

    supported_input_fields = TabularOD.get_compatible_fields(monitored_model.contract.predict.inputs)
    # supported_output_fields = TabularOD.get_compatible_fields(monitored_model.contract.predict.outputs)

    # FIXME. right now there are no outputs in file provided? to reproduce use
    # {"monitored_model_version_id": 83,
    #  "training_data_path": "s3://feature-lake/training-data/83/training_data901590018946752001483.csv"
    #  }

    supported_fields: List[ModelField] = supported_input_fields  # + supported_output_fields
    supported_fields_names: List[str] = [field.name for field in supported_fields]
    supported_fields_dtypes: List[str] = [field.dtype for field in supported_fields]

    logging.info("%s: Reading training data from %s", repr(monitored_model), training_data_path)
    if S3_ENDPOINT:
        s3 = S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT})
        training_data = pd.read_csv(s3.open(training_data_path, mode='rb'))[supported_fields_names]
    else:
        training_data = pd.read_csv(training_data_path)[supported_fields_names]

    # Applying EM-MV

    logging.info("%s: Applying EM-MV", repr(monitored_model))

    chosen_model = model_selection(training_data)
    outlier_detector = chosen_model.recreate(training_data)

    model_status.deploying("Uploading metric to the cluster")
    TrainingStatusStorage.save_status(model_status)

    try:
        # Create temporary directory to copy monitoring model payload there
        # and delete folder later after uploading it to the cluster
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            monitoring_model_folder_path = \
                f"{tmp_dir_name}/{monitored_model.name}v{monitored_model.version}_auto_metric"
            copytree("resources/monitoring_model_template", monitoring_model_folder_path)
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
                                     metadata={
                                         "created_by": "hydro_auto_od",
                                         "training_data_path": training_data_path,
                                         "monitored_model_id": str(monitored_model_version_id),
                                         "monitored_model": repr(monitored_model)
                                     },
                                     install_command="pip install -r requirements.txt",
                                     runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None))

            logging.info("%s: Uploading monitoring model", repr(monitored_model))
            upload_response: UploadResponse = local_model.upload(hs_cluster, wait=True)[local_model]
    except Exception as e:
        logging.exception("%s: Error while uploading monitoring model", repr(monitored_model))
        model_status.failing(f"Failed to pack & deploy monitoring model to a cluster - {str(e)}")
        TrainingStatusStorage.save_status(model_status)
        return -1

    try:
        # Check that this model is found in the cluster
        monitoring_model = ModelVersion.find_by_id(hs_cluster, upload_response.modelversion.id)
    except ModelVersion.NotFound as e:
        logging.exception("%s: Error while finding monitoring model", repr(monitored_model))
        model_status.failing(f"Failed to find deployed monitoring model in a cluster - {str(e)}")
        TrainingStatusStorage.save_status(model_status)
        return -1

    try:
        logging.info("%s: Creating metric spec", repr(monitored_model))
        # Add monitoring model to the monitored model
        metric_config = MetricSpecConfig(monitoring_model.id,
                                         outlier_detector.threshold_,
                                         chosen_model.threshold_comparator)
        MetricSpec.create(hs_cluster, "auto_od_metric", monitored_model.id, metric_config)
    except Exception as e:
        logging.exception("%s: Error while MetricSpec creating", repr(monitored_model))
        model_status.failing(f"Failed to attach deployed monitoring model as a metric - {str(e)}")
        TrainingStatusStorage.save_status(model_status)
        return -1

    model_status.success()
    TrainingStatusStorage.save_status(model_status)

    logging.info("%s: Done", repr(monitored_model))
    return 1