from ..config_parser.test_reporter import Test


class IntelCaffeTest(Test):
    def __init__(self, model, dataset, indep_parameters, dep_parameters):
        super().__init__(model, dataset, indep_parameters, dep_parameters)

    def get_report(self):
        report_res = ('{0};{1};{2};{3};{4};input_shape;{5};{6};Sync;Device: {7}, '
                      'Iteration count: {8}, Thread count: {9}, KMP_AFFINITY: {10}').format(
            self.model.task, self.model.name, self.dataset.name, self.model.source_framework,
            self.indep_parameters.inference_framework, self.model.precision,
            self.indep_parameters.batch_size, self.indep_parameters.device,
            self.indep_parameters.iteration, self.dep_parameters.nthreads,
            self.dep_parameters.kmp_affinity)

        return report_res
