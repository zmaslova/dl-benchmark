from .openvino_process_factory import create_process
from .openvino_test import OpenVINOTest
from ..framework_wrapper import FrameworkWrapper


class OpenVINOWrapper(FrameworkWrapper):
    framework_name = 'OpenVINO DLDT'

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        return create_process(test, executor, log, cpp_benchmark_path)

    @staticmethod
    def create_test(model, dataset, indep_parameters, dep_parameters):
        return OpenVINOTest(model, dataset, indep_parameters, dep_parameters)

    # @staticmethod
    # def get_dependent_parameters_parser():
    #     return OpenVINOParametersParser()
