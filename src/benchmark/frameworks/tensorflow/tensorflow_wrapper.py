from benchmark.frameworks.framework_wrapper import FrameworkWrapper
from benchmark.frameworks.tensorflow.tensorflow_process import TensorFlowProcess
from benchmark.frameworks.tensorflow.tensorflow_test import TensorFlowTest


class TensorFlowWrapper(FrameworkWrapper):
    framework_name = 'TensorFlow'

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        return TensorFlowProcess.create_process(test, executor, log)

    @staticmethod
    def create_test(model, dataset, indep_parameters, dep_parameters):
        return TensorFlowTest(model, dataset, indep_parameters, dep_parameters)

    # @staticmethod
    # def get_dependent_parameters_parser():
    #     return TensorFlowParametersParser()
