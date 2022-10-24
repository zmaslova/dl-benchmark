from src.benchmark.frameworks.intel_caffe.intel_caffe_parameters_parser import IntelCaffeParametersParser
from src.benchmark.frameworks.openvino.openvino_parameters_parser import OpenVINOParametersParser
from src.benchmark.frameworks.tensorflow.tensorflow_parameters_parser import TensorFlowParametersParser


def get_parameters_parser(framework):
    if framework == 'Caffe':
        return IntelCaffeParametersParser()
    elif framework == 'TensorFlow':
        return TensorFlowParametersParser()
    elif framework == 'OpenVINO DLDT':
        return OpenVINOParametersParser()
    else:
        raise NotImplementedError(f'Unknown framework {framework}')