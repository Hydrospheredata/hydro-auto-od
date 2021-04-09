import datetime
import glob
import json
import logging
import os
import joblib
import tempfile
from typing import Tuple
from multiprocessing import Process
from shutil import copytree
from typing import List

import pandas as pd
from s3fs import S3FileSystem

from hydrosdk.modelversion import ModelVersion, ModelVersionBuilder
from hydrosdk.exceptions import BadRequestException, TimeoutException
from hydrosdk.cluster import Cluster
from hydrosdk.image import DockerImage
from hydrosdk.monitoring import ThresholdCmpOp, MetricSpecConfig, MetricSpec
from hydro_serving_grpc.serving.contract.field_pb2 import ModelField

from config import DEFAULT_RUNTIME
from utils import get_monitoring_signature_from_monitored_model, DTYPE_TO_NAMES
from selection import model_selection
from tabular_od_methods import TabularOD
from training_status_storage import TrainingStatusStorage, AutoODMethodStatuses, TrainingStatus
from config import CLUSTER_ENDPOINT, S3_ENDPOINT, DEFAULT_RUNTIME, DEFAULT_TIMEOUT


hs_cluster = Cluster(CLUSTER_ENDPOINT)


def process_auto_metric_request(training_data_path: str, monitored_model_version_id: int) -> Tuple[int, str]:
    logging.info("Started processing auto-od request for modelversion_id=%d", monitored_model_version_id)

    try:
        model_version = ModelVersion.find_by_id(hs_cluster, monitored_model_version_id)
    except BadRequestException as e:
        logging.error(f"{str(e)}")
        return 400, f"Model with modelversion_id={monitored_model_version_id} is not found"

    if TrainingStatusStorage.find_by_model_version_id(monitored_model_version_id) is not None:
        logging.info("A training job is already requested for modelversion_id=%d", monitored_model_version_id)
        return 409, f"A training job is already requested for modelversion_id={monitored_model_version_id}"

    if TabularOD.supports_signature(model_version.signature):
        p = Process(target=train_and_deploy_monitoring_model,
                    args=(monitored_model_version_id, training_data_path))
        p.start()
        logging.info("Started a training job for modelversion_id=%d", monitored_model_version_id)
        return 202, f"Started a training job for modelversion_id={monitored_model_version_id}"
    else:
        logging.warning(
            "Signature of modelversion_id=%d is not supported for creating auto-od metric, aborting", 
            monitored_model_version_id)
        description = (
            "There are 0 supported fields in this model signature. To see how you can support "
            "AutoOD metric refer to the documentation.")
        model_status = TrainingStatus(
            model_version_id=monitored_model_version_id, 
            training_data_path=training_data_path, 
            state=AutoODMethodStatuses.NOT_SUPPORTED, 
            description=description,
        )
        TrainingStatusStorage.save_status(model_status)
        return 200, f"Model state is {model_status.state}, {model_status.description}"


def upload_model_with_circuit_breaker(mv: ModelVersion, times: int=3, timeout: int=120):
    if times <= 0:
        raise TimeoutException
    try:
        mv.lock_till_released(timeout)
    except TimeoutException:
        upload_model_with_circuit_breaker(mv, times=times-1, timeout=timeout*1.5)


def train_and_deploy_monitoring_model(monitored_model_version_id: int, training_data_path: str) -> int:
    """
    This function:
    1. Downloads training data from S3 into pd.Dataframe
    2. Uses this training data to apply EM-MV method to choose a model
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
    model_status = TrainingStatus(
        model_version_id=monitored_model_version_id, 
        training_data_path=training_data_path, 
        state=AutoODMethodStatuses.STARTED, 
        description=description
    )
    TrainingStatusStorage.save_status(model_status)

    logging.info("Retrieving monitored model modelversion_id=%d", monitored_model_version_id)
    monitored_model = ModelVersion.find_by_id(hs_cluster, monitored_model_version_id)

    supported_input_fields = TabularOD.get_compatible_fields(monitored_model.signature.inputs)
    supported_output_fields = TabularOD.get_compatible_fields(monitored_model.signature.outputs)
    supported_fields: List[ModelField] = supported_input_fields + supported_output_fields
    supported_fields_names: List[str] = [field.name for field in supported_fields]

    logging.info(
        "Reading training data from %s for modelversion_id=%d", 
        training_data_path, monitored_model_version_id)
    if S3_ENDPOINT:
        s3 = S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT})
        training_data = pd.read_csv(s3.open(training_data_path, mode='rb', names=supported_fields_names))
    else:
        training_data = pd.read_csv(training_data_path, names=supported_fields_names)

    logging.info(
        "Running outlier model selection algorithm for modelversion_id=%d", monitored_model_version_id)
    outlier_detector = model_selection(training_data)

    logging.info(
        "Selected an outlier model=%s for modelversion_id=%d", 
        repr(outlier_detector), monitored_model_version_id)
    model_status.deploying("Uploading metric to the cluster")
    TrainingStatusStorage.save_status(model_status)

    try:
        # Create temporary directory to copy monitoring model template there
        # and delete folder later after uploading it to the cluster
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            monitoring_model_folder_path = \
                f"{tmp_dir_name}/{monitored_model.name}v{monitored_model.version}_auto_metric"
            copytree("resources/monitoring_model_template", monitoring_model_folder_path)
            joblib.dump(outlier_detector, f'{monitoring_model_folder_path}/outlier_detector.joblib')

            # Save names and dtypes of analysed model fields to use in handling new requests in func_main.py
            monitoring_model_config = {"field_names": supported_fields_names}
            with open(f"{monitoring_model_folder_path}/fields_config.json", "w+") as fields_config_file:
                json.dump(monitoring_model_config, fields_config_file)

            payload_filenames = [os.path.basename(path) for path in glob.glob(f"{monitoring_model_folder_path}/*")]

            model_version_builder = ModelVersionBuilder(monitored_model.name + "_metric", monitoring_model_folder_path) \
                .with_signature(get_monitoring_signature_from_monitored_model(monitored_model)) \
                .with_payload(payload_filenames) \
                .with_runtime(DockerImage.from_string(DEFAULT_RUNTIME)) \
                .with_metadata({
                    "created_by": "hydro_auto_od",
                    "is_metric": 'True',
                    "training_data_path": training_data_path,
                    "monitored_model_id": str(monitored_model_version_id),
                    "monitored_model": repr(monitored_model)
                }) \
                .with_install_command("pip install -r requirements.txt")
            
            logging.info("Uploading a monitoring model for modelversion_id=%d", monitored_model_version_id)
            model_version = model_version_builder.build(hs_cluster)

        upload_model_with_circuit_breaker(model_version, timeout=DEFAULT_TIMEOUT)

    except TimeoutException as e:
        logging.error(
            "Timed out waiting for the outlier detector with modelversion_id=%d to build for "
            "the base model with modelversion_id=%d", model_version.id, monitored_model.id)
        model_status.failing("Monitoring model timed out during model build")
        TrainingStatusStorage.save_status(model_status)
        return -1
    
    except ModelVersion.ReleaseFailed as e:
        logging.error(
            "Outlier detector with modelversion_id=%d failed to build for "
            "the base model with modelversion_id=%d", model_version.id, monitored_model.id)
        model_status.failing("Monitoring model failed to build")
        TrainingStatusStorage.save_status(model_status)
        return -1
        
    except Exception as e:
        logging.exception("Error occurred while uploading an outlier detector for modelversion_id=%d: %s", monitored_model.id, e)
        model_status.failing(f"Failed to pack & deploy monitoring model to a cluster due to: {str(e)}")
        TrainingStatusStorage.save_status(model_status)
        return -1

    try:
        logging.info(
            "Assigning the outlier detector with modelversion_id=%d to the base model with "
            "modelversion_id=%d as metric", model_version.id, monitored_model.id)
        metric = model_version.as_metric(
            threshold=1.0-outlier_detector.contamination, 
            comparator=ThresholdCmpOp.LESS)
        monitored_model.assign_metrics([metric])
    except Exception as e:
        logging.exception("Error occurred while assigning monitoring metric for modelversion_id=%d", monitored_model.id)
        model_status.failing(f"Failed to assign monitoring metrics to the model")
        TrainingStatusStorage.save_status(model_status)
        return -1

    model_status.success()
    TrainingStatusStorage.save_status(model_status)
    logging.info("Finished creating an outlier detector for modelversion_id=%d", monitored_model.id)
    return 1