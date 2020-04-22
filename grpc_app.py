import logging
import os
from concurrent import futures

import grpc
import hydro_serving_grpc as hs_grpc

from app import get_mongo_client, AUTO_OD_DB_NAME, process_auto_metric_request

GRPC_PORT = os.getenv("GRPC_PORT")


class AutoODServiceServicer(hs_grpc.auto_od.AutoOdServiceServicer):
    def __init__(self):
        self.db = get_mongo_client()[AUTO_OD_DB_NAME]

    def GetModelStatus(self, request, context):
        model_status = self.db.model_statuses.find_one({'monitored_model_version_id': request.model_version_id})
        if not model_status:
            return hs_grpc.auto_od.ModelStatus(state=hs_grpc.auto_od.AutoODState.PENDING,
                                               description="Training job for this model version was never requested.")
        else:
            return hs_grpc.auto_od.ModelStatus(state=hs_grpc.auto_od.AutoODState.Value(model_status['state']),
                                               description=model_status['description'])

    def LaunchAutoOd(self, request, context):
        training_data_path, model_version_id = request.training_data_path, request.model_version_id
        code, description = process_auto_metric_request(training_data_path, model_version_id)
        return hs_grpc.auto_od.LauchAutoOdResponse(code=code, description=description)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    hs_grpc.auto_od.add_AutoOdServiceServicer_to_server(AutoODServiceServicer(), server)
    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
