import abc
from types import SimpleNamespace

class TestReport:
    def __init__(self):
        self.task_type
        self.model
        self.dataset
        self.framework
        self.inference_framework
        self.precision
        self.batch_size
        self.mode
        self.framework_params_str

class Test(metaclass=abc.ABCMeta):
    def __init__(self, model, dataset, indep_parameters, dep_parameters):
        self.model = model
        self.dataset = dataset
        self.indep_parameters = indep_parameters
        self.dep_parameters = dep_parameters

    @abc.abstractmethod
    def get_report(self):
        pass
