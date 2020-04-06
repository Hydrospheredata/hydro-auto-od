from hydro_serving_grpc import TensorShapeProto
from hydro_serving_grpc.contract import ModelSignature, ModelField
from hydro_serving_grpc.tf.types_pb2 import *

DTYPE_TO_NAMES = {
    DT_HALF: "float16",
    DT_FLOAT: "float32",
    DT_DOUBLE: "float64",

    DT_INT8: "int8",
    DT_INT16: "int16",
    DT_INT32: "int32",
    DT_INT64: "int64",

    DT_UINT8: "uint8",
    DT_UINT16: "uint16",
    DT_UINT32: "uint32",
    DT_UINT64: "uint64",
}


def get_monitoring_signature_from_monitored_signature(monitored_model_signature):
    concatenated_input = list(monitored_model_signature.inputs) + list(monitored_model_signature.outputs)
    monitoring_model_signature = ModelSignature(inputs=concatenated_input,
                                                outputs=[ModelField(name="value",
                                                                    shape=TensorShapeProto(),
                                                                    dtype=DT_DOUBLE)])
    return monitoring_model_signature
