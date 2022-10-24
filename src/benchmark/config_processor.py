from src.benchmark.config_parser.test_config_parser import TestConfigParser
from src.benchmark.frameworks.framework_wrapper_registry import FrameworkWrapperRegistry

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
