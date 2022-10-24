import os

from benchmark.frameworks.openvino.openvino_benchmark_process import OpenVINOBenchmarkProcess


class OpenVINOBenchmarkCppProcess(OpenVINOBenchmarkProcess):
    def __init__(self, test, executor, log, cpp_benchmark_path, perf_hint=''):
        super().__init__(test, executor, log, perf_hint)
        self._benchmark_path = cpp_benchmark_path
        self._perf_hint = perf_hint

        if not cpp_benchmark_path or not os.path.exists(cpp_benchmark_path):
            raise ValueError('Must provide valid cpp_benchmark_path for OpenVINO C++ benchmark')

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        return OpenVINOBenchmarkCppProcess(test, executor, log, cpp_benchmark_path)

    def _fill_command_line(self):
        model_xml = self._test.model.model
        dataset = self._test.dataset.path
        batch = self._test.indep_parameters.batch_size
        device = self._test.indep_parameters.device
        iteration = self._test.indep_parameters.iteration

        arguments = f'-m {model_xml} -i {dataset} -b {batch} -d {device} -niter {iteration} -report_type "no_counters"'

        extension = self._test.dep_parameters.extension
        if extension:
            arguments = self._add_extension_for_cmd_line(arguments, extension, device)

        nthreads = self._test.dep_parameters.nthreads
        if nthreads:
            arguments = self.__add_nthreads_for_cmd_line(arguments, nthreads)

        arguments = self._add_perf_hint_for_cmd_line(arguments, self._perf_hint)

        command_line = f'{self._benchmark_path} {arguments}'
        return command_line
