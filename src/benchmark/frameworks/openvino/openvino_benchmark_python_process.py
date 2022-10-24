from .openvino_benchmark_process import OpenVINOBenchmarkProcess


class OpenVINOBenchmarkPythonProcess(OpenVINOBenchmarkProcess):
    def __init__(self, test, executor, log, perf_hint=''):
        super().__init__(test, executor, log, perf_hint)
        self._perf_hint = perf_hint

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        return OpenVINOBenchmarkPythonProcess(test, executor, log)

    def _fill_command_line(self):
        model_xml = self._test.model.model
        dataset = self._test.dataset.path
        batch = self._test.indep_parameters.batch_size
        device = self._test.indep_parameters.device
        iteration = self._test.indep_parameters.iteration

        arguments = f'-m {model_xml} -i {dataset} -b {batch} -d {device} -niter {iteration}'

        extension = self._test.dep_parameters.extension
        if extension:
            arguments = self._add_extension_for_cmd_line(arguments, extension, device)

        nthreads = self._test.dep_parameters.nthreads
        if nthreads:
            arguments = self.__add_nthreads_for_cmd_line(arguments, nthreads)

        arguments = self._add_perf_hint_for_cmd_line(arguments, self._perf_hint)

        command_line = f'benchmark_app {arguments}'
        return command_line
