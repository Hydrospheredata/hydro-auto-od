from hydro_serving_grpc import TensorShapeProto
from hydro_serving_grpc.contract import ModelSignature, ModelField


def get_monitoring_signature_from_monitored_signature(monitored_model_signature):
    concatenated_input = list(monitored_model_signature.inputs) + list(monitored_model_signature.outputs)
    monitoring_model_signature = ModelSignature(inputs=concatenated_input,
                                                outputs=[ModelField(name="metric_value", shape=TensorShapeProto())])
    return monitoring_model_signature
