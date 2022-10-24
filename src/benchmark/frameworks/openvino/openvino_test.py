from collections import OrderedDict

from ..config_parser.test_reporter import Test


class OpenVINOTest(Test):
    def __init__(self, model, dataset, indep_parameters, dep_parameters):
        super().__init__(model, dataset, indep_parameters, dep_parameters)

    def get_report(self):
        parameters = OrderedDict()
        parameters.update({'Device': self.indep_parameters.device})
        parameters.update({'Async request count': self.dep_parameters.async_request})
        parameters.update({'Iteration count': self.indep_parameters.iteration})
        parameters.update({'Thread count': self.dep_parameters.nthreads})
        parameters.update({'Stream count': self.dep_parameters.nstreams})
        other_param = []
        for key in parameters:
            if parameters[key] is not None:
                other_param.append(f'{key}: {parameters[key]}')
        other_param = ', '.join(other_param)

        report_res = '{0};{1};{2};{3};{4};input_shape;{5};{6};{7};{8}'.format(
            self.model.task, self.model.name, self.dataset.name, self.model.source_framework,
            self.indep_parameters.inference_framework, self.model.precision,
            self.indep_parameters.batch_size, self.dep_parameters.mode, other_param)

        return report_res
