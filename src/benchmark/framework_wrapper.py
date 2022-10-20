import importlib
import inspect
import pkgutil
from abc import ABCMeta, abstractmethod

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FrameworkWrapper(metaclass=ABCMeta):
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
    def create_test_result(model, dataset, indep_parameters, dep_parameters):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_dependent_parameters_parser():
        raise NotImplementedError()


class FrameworkWrapperManager(metaclass=Singleton):
    def __init__(self):
        self._framework_wrappers = {}
        self._find_wrappers('frameworks')

    def __getitem__(self, framework_name):
        from config_parser import Test
        if isinstance(framework_name, Test):
            framework_name = framework_name.indep_parameters.inference_framework
        
        if framework_name in self._framework_wrappers:
            return self._framework_wrappers[framework_name]
        else:
            raise ValueError(f'Invalid framework name. Supported values: "{", ".join(self._framework_wrappers.keys())}"')
        
    def _find_wrappers(self, wrappers_path):
        wrappers_package = importlib.import_module(wrappers_path)
        for module_info in pkgutil.iter_modules(wrappers_package.__path__):
            if not module_info.ispkg:
                wrapper = importlib.import_module(f'{wrappers_package.__name__}.{module_info.name}')
                classes = inspect.getmembers(wrapper, inspect.isclass)
                for _, type in classes:
                    if issubclass(type, FrameworkWrapper) and type is not FrameworkWrapper:
                        self._framework_wrappers[type.framework_name] = type()
