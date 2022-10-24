from .frameworks.intel_caffe import IntelCaffeParametersParser
from .frameworks.openvino.openvino_parameters_parser import OpenVINOParametersParser
from .frameworks.tensorflow.tensorflow_parameters_parser import TensorFlowParametersParser


def get_parameters_parser(framework):
    if framework == 'Caffe':
        return IntelCaffeParametersParser()
    elif framework == 'TensorFlow':
        return TensorFlowParametersParser()
    elif framework == 'OpenVINO DLDT':
        return OpenVINOParametersParser()
    else:
        raise NotImplementedError(f'Unknown framework {framework}')