import logging as log

from src.benchmark.frameworks.Singleton import Singleton
from src.benchmark.frameworks.intel_caffe.intel_caffe_wrapper import IntelCaffeWrapper
from src.benchmark.frameworks.openvino.openvino_wrapper import OpenVINOWrapper
from src.benchmark.frameworks.tensorflow.tensorflow_wrapper import TensorFlowWrapper


class FrameworkWrapperRegistry(metaclass=Singleton):
    """Storage for all found framework wrappers.
    Framework wrapper is represented by a FrameworkWrapper subclass located in
    a separate module (openvino.py, tensorflow.py etc) inside frameworks package.
    """

    def __init__(self):
        self._framework_wrappers = {}
        self._find_wrappers('frameworks')
        log.info(f'Available framework wrappers: {", ".join(self._framework_wrappers.keys())}')

    def __getitem__(self, framework_name):
        """Get framework wrapper by framework name"""
        if framework_name in self._framework_wrappers:
            return self._framework_wrappers[framework_name]
        else:
            raise ValueError(f'Unsupported framework name: {framework_name}. '
                             f'Available framework wrappers: {", ".join(self._framework_wrappers.keys())}')

    # def _find_wrappers(self, wrappers_pkg_name):
    #     """Search framework wrappers inside the specified package"""
    #     wrappers_package = importlib.import_module(wrappers_pkg_name)
    #     for module_info in pkgutil.iter_modules(wrappers_package.__path__):
    #         if not module_info.ispkg:
    #             module = importlib.import_module(f'{wrappers_package.__name__}.{module_info.name}')
    #             for _, class_type in inspect.getmembers(module, inspect.isclass):
    #                 if issubclass(class_type, FrameworkWrapper) and class_type is not FrameworkWrapper:
    #                     self._framework_wrappers[class_type.framework_name] = class_type()
    def _find_wrappers(self):
        self._framework_wrappers[IntelCaffeWrapper.framework_name] = IntelCaffeWrapper()
        self._framework_wrappers[TensorFlowWrapper.framework_name] = TensorFlowWrapper()
        self._framework_wrappers[OpenVINOWrapper.framework_name] = OpenVINOWrapper()
