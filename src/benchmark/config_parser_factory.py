from frameworks.intel_caffe.intel_caffe_parameters_parser import IntelCaffeParametersParser
from frameworks.known_frameworks import KnownFrameworks
from frameworks.openvino.openvino_parameters_parser import OpenVINOParametersParser
from frameworks.tensorflow.tensorflow_parameters_parser import TensorFlowParametersParser


def get_parameters_parser(framework):
    if framework == KnownFrameworks.caffe:
        return IntelCaffeParametersParser()
    elif framework == KnownFrameworks.tensorflow:
        return TensorFlowParametersParser()
    elif framework == KnownFrameworks.opevino_dldt:
        return OpenVINOParametersParser()
    else:
        raise NotImplementedError(f'Unknown framework {framework}')