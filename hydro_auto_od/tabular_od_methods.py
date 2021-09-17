from abc import ABC
from typing import List
from hydro_serving_grpc.serving.contract.signature_pb2 import ModelSignature
from hydro_serving_grpc.serving.contract.field_pb2 import ModelField
from hydro_serving_grpc.serving.contract.types_pb2 import (
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, 
    DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_STRING
) 


class TabularOD(ABC):
    _SUPPORTED_DTYPES = {DT_HALF, DT_FLOAT, DT_DOUBLE,
                         DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                         DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_STRING}

    @classmethod
    def get_compatible_fields(cls, inputs: List[ModelField]) -> List[ModelField]:
        def is_tensor_compatible(field: ModelField) -> bool:
            return len(field.shape.dims) == 0 and field.dtype in cls._SUPPORTED_DTYPES
        return list(filter(is_tensor_compatible, inputs))

    @classmethod
    def supports_signature(cls, signature: ModelSignature) -> bool:
        compatible_input_fields = cls.get_compatible_fields(signature.inputs)
        compatible_output_fields = cls.get_compatible_fields(signature.inputs)
        return len(compatible_input_fields) + len(compatible_output_fields) > 0

