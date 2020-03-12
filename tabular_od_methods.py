from typing import Callable
from pyod.models.hbos import HBOS
import hydro_serving_grpc as hs

class TabularOutlierDetectionMethod:

	@staticmethod
	def from_id(method_id):
		if method_id == 1:
			return AutoHBOS()

	def predict(self):
		pass

	@classmethod
	def get_compatible_tensors(cls, inputs):
		NUMERICAL_DTYPES = [9]  # Why only 9?

		def is_tensor_compatible(tensor):
			return (len(tensor.shape.dim) == 0 and tensor.dtype in NUMERICAL_DTYPES)  # 2 stands for 'NUMERICAL' profile

		tensors = list(filter(is_tensor_compatible, inputs))
		# tensors = [{'name': t.name, 'dtype': t.dtype, 'profile': t.profile} for t in tensors]
		tensors = [{'name': t.name, 'dtype': 'DT_INT64', 'profile': 'NUMERICAL', 'shape': {'dim':[], 'unknownRank':False}} for t in tensors]
		return sorted(tensors, key=lambda i: i['name'])

	@classmethod
	def format_output_tensors(cls, outputs):
		tensors = list(outputs)

		# TODO Use this enum to cast from numbers to strings
		#  https://github.com/Hydrospheredata/hydro-serving-protos/blob/master/src/hydro_serving_grpc/tf/types.proto
		
		
		tensors = [{'name': t.name, 'dtype': hs.DataType.Name(t.dtype), 'profile': 'NUMERICAL', 'shape': {'dim':[], 'unknownRank':False}} for t in tensors]
		return sorted(tensors, key=lambda i: i['name'])

	def launch(self):
		pass


class AutoHBOS(TabularOutlierDetectionMethod):
	def __init__(self):
		self.name = "hbos"
		self.id = 1
		self.comment = ""
		self._hbos = HBOS()
		self.requirements = "scikit-learn==0.20.2\nnumpy==1.16.2"
		self.predict: Callable = self._hbos.predict
		self.fit: Callable = self._hbos.fit

# self.requirements = ['scikit-learn==0.20.2',
#                                'pickle',
#                                'numpy==1.16.2']