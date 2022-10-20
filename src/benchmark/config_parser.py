import abc
from xml.dom import minidom

from framework_wrapper import FrameworkWrapperRegistry


class TestConfigParser:
    def get_tests_list(self, config):
        CONFIG_ROOT_TAG = 'Test'
        return minidom.parse(config).getElementsByTagName(CONFIG_ROOT_TAG)

    def parse_model(self, curr_test):
        CONFIG_MODEL_TAG = 'Model'
        CONFIG_MODEL_TASK_TAG = 'Task'
        CONFIG_MODEL_NAME_TAG = 'Name'
        CONFIG_MODEL_PRECISION_TAG = 'Precision'
        CONFIG_MODEL_SOURCE_FRAMEWORK_TAG = 'SourceFramework'
        CONFIG_MODEL_MODEL_PATH_TAG = 'ModelPath'
        CONFIG_MODEL_WEIGHTS_PATH_TAG = 'WeightsPath'

        model_tag = curr_test.getElementsByTagName(CONFIG_MODEL_TAG)[0]

        return Model(
            task=model_tag.getElementsByTagName(CONFIG_MODEL_TASK_TAG)[0].firstChild.data,
            name=model_tag.getElementsByTagName(CONFIG_MODEL_NAME_TAG)[0].firstChild.data,
            precision=model_tag.getElementsByTagName(CONFIG_MODEL_PRECISION_TAG)[0].firstChild.data,
            source_framework=model_tag.getElementsByTagName(CONFIG_MODEL_SOURCE_FRAMEWORK_TAG)[0].firstChild.data,
            model_path=model_tag.getElementsByTagName(CONFIG_MODEL_MODEL_PATH_TAG)[0].firstChild.data,
            weights_path=model_tag.getElementsByTagName(CONFIG_MODEL_WEIGHTS_PATH_TAG)[0].firstChild.data,
        )

    def parse_dataset(self, curr_test):
        CONFIG_DATASET_TAG = 'Dataset'
        CONFIG_DATASET_NAME_TAG = 'Name'
        CONFIG_DATASET_PATH_TAG = 'Path'

        dataset_tag = curr_test.getElementsByTagName(CONFIG_DATASET_TAG)[0]

        return Dataset(
            name=dataset_tag.getElementsByTagName(CONFIG_DATASET_NAME_TAG)[0].firstChild.data,
            path=dataset_tag.getElementsByTagName(CONFIG_DATASET_PATH_TAG)[0].firstChild.data,
        )

    def parse_independent_parameters(self, curr_test):
        CONFIG_FRAMEWORK_INDEPENDENT_TAG = 'FrameworkIndependent'
        CONFIG_FRAMEWORK_INDEPENDENT_INFERENCE_FRAMEWORK_TAG = 'InferenceFramework'
        CONFIG_FRAMEWORK_INDEPENDENT_BATCH_SIZE_TAG = 'BatchSize'
        CONFIG_FRAMEWORK_INDEPENDENT_DEVICE_TAG = 'Device'
        CONFIG_FRAMEWORK_INDEPENDENT_ITERATION_COUNT_TAG = 'IterationCount'
        CONFIG_FRAMEWORK_INDEPENDENT_TEST_TIME_LIMIT_TAG = 'TestTimeLimit'

        indep_parameters_tag = curr_test.getElementsByTagName(CONFIG_FRAMEWORK_INDEPENDENT_TAG)[0]

        return FrameworkIndependentParameters(
            inference_framework=indep_parameters_tag.getElementsByTagName(
                CONFIG_FRAMEWORK_INDEPENDENT_INFERENCE_FRAMEWORK_TAG)[0].firstChild.data,
            batch_size=indep_parameters_tag.getElementsByTagName(
                CONFIG_FRAMEWORK_INDEPENDENT_BATCH_SIZE_TAG)[0].firstChild.data,
            device=indep_parameters_tag.getElementsByTagName(
                CONFIG_FRAMEWORK_INDEPENDENT_DEVICE_TAG)[0].firstChild.data,
            iterarion_count=indep_parameters_tag.getElementsByTagName(
                CONFIG_FRAMEWORK_INDEPENDENT_ITERATION_COUNT_TAG)[0].firstChild.data,
            test_time_limit=indep_parameters_tag.getElementsByTagName(
                CONFIG_FRAMEWORK_INDEPENDENT_TEST_TIME_LIMIT_TAG)[0].firstChild.data,
        )

    def parse_dependent_parameters(self, curr_test, framework):
        dep_parser = FrameworkWrapperRegistry()[framework].get_dependent_parameters_parser()
        return dep_parser.parse_parameters(curr_test)


class DependentParametersParser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def parse_parameters(self, curr_test):
        pass


class Model:
    def __init__(self, task, name, model_path, weights_path, precision, source_framework):
        """
        :param task:
        :type task:
        :param name:
        :type name:
        :param model_path:
        :type model_path:
        :param weights_path:
        :type weights_path:
        :param precision:
        :type precision:
        :param source_framework:
        :type source_framework:
        """
        self.source_framework = None
        self.task = task
        self.name = None
        self.model = None
        self.weight = None
        self.precision = None
        if self._parameter_not_is_none(source_framework):
            self.source_framework = source_framework
        else:
            raise ValueError('Source framework is required parameter.')
        if self._parameter_not_is_none(name):
            self.name = name
        else:
            raise ValueError('Model name is required parameter.')
        if self._parameter_not_is_none(model_path):
            self.model = model_path
        else:
            raise ValueError('Path to model is required parameter.')
        if self._parameter_not_is_none(weights_path):
            self.weight = weights_path
        else:
            raise ValueError('Path to model weights is required parameter.')
        if self._parameter_not_is_none(precision):
            self.precision = precision
        else:
            raise ValueError('Precision is required parameter.')

    @staticmethod
    def _parameter_not_is_none(parameter):
        return True if parameter is not None else False


class Dataset:
    def __init__(self, name, path):
        self.name = None
        self.path = None
        if self._parameter_not_is_none(name):
            self.name = name
        else:
            raise ValueError('Dataset name is required parameter.')
        if self._parameter_not_is_none(path):
            self.path = path
        else:
            raise ValueError('Path to dataset is required parameter.')

    @staticmethod
    def _parameter_not_is_none(parameter):
        return True if parameter is not None else False


class Parameters:
    @staticmethod
    def _parameter_not_is_none(parameter):
        return True if parameter is not None else False

    @staticmethod
    def _int_value_is_correct(int_value):
        for i in range(len(int_value)):
            if (i < 0) or (9 < i):
                return False
        return True

    def _float_value_is_correct(self, float_value):
        for i in float_value.split('.'):
            if not self._int_value_is_correct(i):
                return False
        return True


class FrameworkIndependentParameters(Parameters):
    def __init__(self, inference_framework, batch_size, device, iterarion_count, test_time_limit):
        self.inference_framework = None
        self.batch_size = None
        self.device = None
        self.iteration = None
        self.test_time_limit = None
        if self._parameter_not_is_none(inference_framework):
            self.inference_framework = inference_framework
        else:
            raise ValueError('Inference framework is required parameter.')
        if self._parameter_not_is_none(batch_size) and self._int_value_is_correct(batch_size):
            self.batch_size = int(batch_size)
        else:
            raise ValueError('Batch size is required parameter. '
                             'Batch size can only take values: integer greater than zero.')
        if self._device_is_correct(device):
            self.device = device.upper()
        else:
            raise ValueError('Device is required parameter. '
                             'Supported values: CPU, GPU, FPGA, MYRIAD.')
        if self._parameter_not_is_none(iterarion_count) and self._int_value_is_correct(iterarion_count):
            self.iteration = int(iterarion_count)
        else:
            raise ValueError('Iteration count is required parameter. '
                             'Iteration count can only take values: integer greater than zero.')
        if self._parameter_not_is_none(test_time_limit) and self._float_value_is_correct(test_time_limit):
            self.test_time_limit = float(test_time_limit)
        else:
            raise ValueError('Test time limit is required parameter. '
                             'Test time limit can only `take values: float greater than zero.')

    @staticmethod
    def _device_is_correct(device):
        const_correct_devices = ['CPU', 'GPU', 'MYRIAD', 'FPGA']
        if device.upper() in const_correct_devices:
            return True
        return False


class Test(metaclass=abc.ABCMeta):
    def __init__(self, model, dataset, indep_parameters, dep_parameters):
        self.model = model
        self.dataset = dataset
        self.indep_parameters = indep_parameters
        self.dep_parameters = dep_parameters

    @abc.abstractmethod
    def get_report(self):
        pass


def process_config(config, log):
    test_parser = TestConfigParser()
    test_list = []

    tests = test_parser.get_tests_list(config)
    for idx, curr_test in enumerate(tests):
        try:
            model = test_parser.parse_model(curr_test)
            dataset = test_parser.parse_dataset(curr_test)
            indep_parameters = test_parser.parse_independent_parameters(curr_test)
            framework = indep_parameters.inference_framework
            dep_parameters = test_parser.parse_dependent_parameters(curr_test, framework)

            test_list.append(FrameworkWrapperRegistry()[framework].create_test(model, dataset,
                                                                               indep_parameters, dep_parameters))
        except ValueError as valerr:
            log.warning(f'Test {idx + 1} not added to test list: {valerr}')
    return test_list
