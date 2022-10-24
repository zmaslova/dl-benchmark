import logging as log

from .Singleton import Singleton
from .intel_caffe.intel_caffe_wrapper import IntelCaffeWrapper
from .openvino.openvino_wrapper import OpenVINOWrapper
from .tensorflow.tensorflow_wrapper import TensorFlowWrapper


class FrameworkWrapperRegistry(metaclass=Singleton):
    """Storage for all found framework wrappers.
    Framework wrapper is represented by a FrameworkWrapper subclass located in
    a separate module (openvino.py, tensorflow.py etc) inside frameworks package.
    """

    def __init__(self):
        self._framework_wrappers = {}
        self._find_wrappers()
        log.info(f'Available framework wrappers: {", ".join(self._framework_wrappers.keys())}')

    def __getitem__(self, framework_name):
        """Get framework wrapper by framework name"""
        if framework_name in self._framework_wrappers:
            return self._framework_wrappers[framework_name]
        else:
            raise ValueError(f'Unsupported framework name: {framework_name}. '
                             f'Available framework wrappers: {", ".join(self._framework_wrappers.keys())}')

    def _find_wrappers(self):
        self._framework_wrappers[IntelCaffeWrapper.framework_name] = IntelCaffeWrapper()
        self._framework_wrappers[TensorFlowWrapper.framework_name] = TensorFlowWrapper()
        self._framework_wrappers[OpenVINOWrapper.framework_name] = OpenVINOWrapper()
