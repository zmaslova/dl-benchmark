from benchmark.frameworks.framework_wrapper import FrameworkWrapper
from benchmark.frameworks.openvino.openvino_process_factory import create_process
from benchmark.frameworks.openvino.openvino_test import OpenVINOTest


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
