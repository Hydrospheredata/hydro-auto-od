from abc import ABC
from typing import Callable, List

from hydro_serving_grpc.contract import ModelSignature, ModelField
from hydro_serving_grpc.tf.types_pb2 import *
from pyod.models.hbos import HBOS


class TabularOD(ABC):
    _SUPPORTED_DTYPES = {DT_HALF, DT_FLOAT, DT_DOUBLE,
                         DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                         DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}

    @staticmethod
    def from_id(method_id):
        if method_id == 1:
            return AutoHBOS()

    @classmethod
    def get_compatible_fields(cls, inputs: List[ModelField]) -> List[ModelField]:
        def is_tensor_compatible(tensor):
            return len(tensor.shape.dim) == 0 and tensor.dtype in cls._SUPPORTED_DTYPES

        return list(filter(is_tensor_compatible, inputs))

    @classmethod
    def supports_signature(cls, signature: ModelSignature) -> bool:
        compatible_input_fields = cls.get_compatible_fields(signature.inputs)
        compatible_output_fields = cls.get_compatible_fields(signature.inputs)
        return len(compatible_input_fields) + len(compatible_output_fields) > 0


class AutoHBOS(TabularOD):
    def __init__(self):
        self.name = "HBOS"
        self.id = 1
        self.comment = ""
        self._hbos = HBOS()
        self.requirements = "scikit-learn==0.20.2\nnumpy==1.16.2"
        self.predict: Callable = self._hbos.predict
        self.fit: Callable = self._hbos.fit
