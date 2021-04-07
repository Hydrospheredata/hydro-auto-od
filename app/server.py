import logging
from logging.config import fileConfig
import os
from concurrent import futures

import grpc
from hydro_serving_grpc.monitoring.auto_od.api_pb2 import (
    ModelStatusRequest, ModelStatusResponse, LaunchAutoOdRequest,
    LaunchAutoOdResponse, AutoODState
)
from hydro_serving_grpc.monitoring.auto_od.api_pb2_grpc import (
    AutoOdServiceServicer, add_AutoOdServiceServicer_to_server
)

from grpc_health.v1.health_pb2 import HealthCheckResponse
from grpc_health.v1.health_pb2_grpc import HealthServicer
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server

from app.main import process_auto_metric_request
from app.config import GRPC_PORT
from app.training_status_storage import TrainingStatusStorage

fileConfig("resources/logging_config.ini")


class AutoODServiceServicer(AutoOdServiceServicer, HealthServicer):
    def GetModelStatus(self, request: ModelStatusRequest, context):
        model_status = TrainingStatusStorage.find_by_model_version_id(request.model_version_id)
        if model_status is not None:
            return ModelStatusResponse(
                state=AutoODState.Value(model_status.state),
                description=model_status.description
            )
        else:
            return ModelStatusResponse(
                state=AutoODState.PENDING,
                description=f"Training job for modelversion_id={request.model_version_id} was never requested."
            )

    def LaunchAutoOd(self, request: LaunchAutoOdRequest, context):
        training_data_path, model_version_id = request.training_data_path, request.model_version_id
        state, description = process_auto_metric_request(training_data_path, model_version_id)
        return LaunchAutoOdResponse(state=state, description=description)

    def Check(self, request, context):
        return HealthCheckResponse(status="SERVING")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    servicer = AutoODServiceServicer()
    add_AutoOdServiceServicer_to_server(servicer, server)
    add_HealthServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    server.start()
    logging.info(f"Server started at [::]:{GRPC_PORT}")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
