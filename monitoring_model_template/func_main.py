import hydro_serving_grpc as hs
import numpy as np
import pickle
# model = load('/model/files/HBOS.model')
model = pickle.load(open('/model/files/hbos.model', 'rb'))


def extract_value(proto):
    return np.array(proto.int64_val, dtype='int64').reshape([dim.size for dim in proto.tensor_shape.dim])


def predict(**kwargs):

    # TODO fix it!
    x = extract_value(kwargs['input'])
    predicted = model.predict(x)

    response = hs.TensorProto(
        int64_val=predicted.flatten().tolist(),
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1)]))

    return hs.PredictResponse(outputs={"classes": response})