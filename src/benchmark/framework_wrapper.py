import importlib
import inspect
import logging as log
import pkgutil
from abc import ABCMeta, abstractmethod


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FrameworkWrapper(metaclass=ABCMeta):
    """Base abstract class for framework wrapper.
    The framework_name attribute should be initialized in a derived class
    with framework name string used in configuration file."""

    framework_name = ''

    def __init_subclass__(cls):
        if not isinstance(cls.framework_name, str) or len(cls.framework_name) == 0:
            raise NotImplementedError(f'Static attribute framework_name is not initialized in class {cls.__name__}')

    @staticmethod
    @abstractmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def create_test(model, dataset, indep_parameters, dep_parameters):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_dependent_parameters_parser():
        raise NotImplementedError()


class FrameworkWrapperRegistry(metaclass=Singleton):
    """Storage for all available framework wrappers.
    A framework wrapper is `FrameworkWrapper` subclass located in a separate module (openvino.py, tensorflow.py etc)
    inside frameworks/ package
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

    def _find_wrappers(self, wrappers_pkg_name):
        """Search framework wrappers inside the specified package"""
        wrappers_package = importlib.import_module(wrappers_pkg_name)
        for module_info in pkgutil.iter_modules(wrappers_package.__path__):
            if not module_info.ispkg:
                module = importlib.import_module(f'{wrappers_package.__name__}.{module_info.name}')
                for _, class_type in inspect.getmembers(module, inspect.isclass):
                    if issubclass(class_type, FrameworkWrapper) and class_type is not FrameworkWrapper:
                        self._framework_wrappers[class_type.framework_name] = class_type()
