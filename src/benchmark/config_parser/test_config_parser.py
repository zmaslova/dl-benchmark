from xml.dom import minidom

from benchmark.config_parser_factory import get_parameters_parser
from src.benchmark.config_parser.dataset_parser import Dataset
from src.benchmark.config_parser.framework_independent_parameters import FrameworkIndependentParameters
from src.benchmark.config_parser.model import Model


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
        dep_parser = get_parameters_parser(framework)
        return dep_parser.parse_parameters(curr_test)
