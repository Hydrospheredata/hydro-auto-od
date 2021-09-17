from hydrosdk.modelversion import ModelVersion
from hydro_serving_grpc.serving.contract.tensor_pb2 import TensorShape
from hydro_serving_grpc.serving.contract.signature_pb2 import ModelSignature
from hydro_serving_grpc.serving.contract.field_pb2 import ModelField
from hydro_serving_grpc.serving.contract.types_pb2 import (
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, 
    DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64
)

DTYPE_TO_NAMES = {
    DT_HALF: "half",
    DT_FLOAT: "float",
    DT_DOUBLE: "double",

    DT_INT8: "int8",
    DT_INT16: "int16",
    DT_INT32: "int32",
    DT_INT64: "int64",

    DT_UINT8: "uint8",
    DT_UINT16: "uint16",
    DT_UINT32: "uint32",
    DT_UINT64: "uint64",
}


def get_monitoring_signature_from_monitored_model(model_version: ModelVersion) -> ModelSignature:
    concatenated_input = list(model_version.signature.inputs) + list(model_version.signature.outputs)
    monitoring_model_signature = ModelSignature(signature_name="predict",
                                                inputs=concatenated_input,
                                                outputs=[ModelField(name="value",
                                                                    shape=TensorShape(),
                                                                    dtype=DT_DOUBLE)])
    return monitoring_model_signature
