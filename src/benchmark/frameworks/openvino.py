import os
import re
from abc import ABC
from collections import OrderedDict

from config_parser import DependentParametersParser, ParametersMethods, Test
from framework_wrapper import FrameworkWrapper
from processes import ProcessHandler


class OpenVINOWrapper(FrameworkWrapper):
    framework_name = 'OpenVINO DLDT'

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        return OpenVINOProcess.create_process(test, executor, log, cpp_benchmark_path)

    @staticmethod
    def create_test_result(model, dataset, indep_parameters, dep_parameters):
        return OpenVINOTest(model, dataset, indep_parameters, dep_parameters)

    @staticmethod
    def get_dependent_parameters_parser():
        return OpenVINOParametersParser()


class OpenVINOProcess(ProcessHandler, ABC):
    def __init__(self, test, executor, log):
        super().__init__(test, executor, log)

    @staticmethod
    def __add_nthreads_for_cmd_line(command_line, nthreads):
        return f'{command_line} -nthreads {nthreads}'

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        mode = test.dep_parameters.mode.lower()
        if mode == 'sync':
            return SyncOpenVINOProcess(test, executor, log)
        elif mode == 'async':
            return AsyncOpenVINOProcess(test, executor, log)
        elif mode == 'ovbenchmark_python_latency':
            return OpenVINOBenchmarkPythonProcess(test, executor, log, 'latency')
        elif mode == 'ovbenchmark_python_throughput':
            return OpenVINOBenchmarkPythonProcess(test, executor, log, 'throughput')
        elif mode == 'ovbenchmark_cpp_latency':
            return OpenVINOBenchmarkCppProcess(test, executor, log, cpp_benchmark_path, 'latency')
        elif mode == 'ovbenchmark_cpp_throughput':
            return OpenVINOBenchmarkCppProcess(test, executor, log, cpp_benchmark_path, 'throughput')


class OpenVINOBenchmarkProcess(OpenVINOProcess):
    def __init__(self, test, executor, log, perf_hint=''):
        super().__init__(test, executor, log)
        self._perf_hint = perf_hint

    @staticmethod
    def _add_perf_hint_for_cmd_line(command_line, perf_hint):
        hint = perf_hint.lower()
        if hint in ('latency', 'throughput'):
            return f'{command_line} -hint {hint}'
        return command_line

    @staticmethod
    def _add_extension_for_cmd_line(command_line, extension, device):
        if 'GPU' in device:
            return f'{command_line} -c {extension}'
        elif 'CPU' in device or 'MYRIAD' in device:
            return f'{command_line} -l {extension}'
        return command_line

    def get_performance_metrics(self):
        if self._status != 0 or len(self._output) == 0:
            return None, None, None

        # calculate average time of single pass metric to align output with custom launchers
        duration = self._get_benchmark_app_metric('Duration')
        iter_count = self._get_benchmark_app_metric('Count')
        average_time_of_single_pass = (round(duration / 1000 / iter_count, 3)
                                       if None not in (duration, iter_count) else None)

        fps = self._get_benchmark_app_metric('Throughput')
        latency = round(self._get_benchmark_app_metric('Median') / 1000, 3)

        return average_time_of_single_pass, fps, latency

    def _get_benchmark_app_metric(self, metric_name):
        """
        gets metric value from benchmark app full output
        :param metric_name: metric name, ex 'Throughput'
        :return: float value or None if pattern not found
        """
        for line in self._output:
            regex = re.compile(f'.*{metric_name}:\\s+(?P<metric>\\d*\\.\\d+|\\d+).*')
            res = regex.match(line)
            if res:
                try:
                    return float(res.group('metric'))
                except ValueError:
                    return None


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


class OpenVINOPythonAPIProcess(OpenVINOProcess):
    def __init__(self, test, executor, log):
        super().__init__(test, executor, log)

    @staticmethod
    def __add_extension_for_cmd_line(command_line, extension):
        return f'{command_line} -l {extension}'

    @staticmethod
    def __add_raw_output_time_for_cmd_line(command_line, raw_output):
        return f'{command_line} {raw_output}'

    def _fill_command_line(self):
        model_xml = self._test.model.model
        model_bin = self._test.model.weight
        dataset = self._test.dataset.path
        batch = self._test.indep_parameters.batch_size
        device = self._test.indep_parameters.device
        iteration = self._test.indep_parameters.iteration

        command_line = f'-m {model_xml} -w {model_bin} -i {dataset} -b {batch} -d {device} -ni {iteration}'

        extension = self._test.dep_parameters.extension
        if extension:
            command_line = OpenVINOPythonAPIProcess.__add_extension_for_cmd_line(command_line, extension)
        nthreads = self._test.dep_parameters.nthreads
        if nthreads:
            command_line = OpenVINOPythonAPIProcess.__add_nthreads_for_cmd_line(command_line, nthreads)
        command_line = OpenVINOPythonAPIProcess.__add_raw_output_time_for_cmd_line(command_line, '--raw_output true')

        return command_line


class SyncOpenVINOProcess(OpenVINOPythonAPIProcess):
    def __init__(self, test, executor, log):
        super().__init__(test, executor, log)

    def get_performance_metrics(self):
        if self._status != 0 or len(self._output) == 0:
            return None, None, None

        result = self._output[-1].strip().split(',')
        average_time = float(result[0])
        fps = float(result[1])
        latency = float(result[2])

        return average_time, fps, latency

    def _fill_command_line(self):
        path_to_sync_scrypt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'inference',
                                           'inference_sync_mode.py')
        python = ProcessHandler._get_cmd_python_version()

        common_params = super()._fill_command_line()
        command_line = f'{python} {path_to_sync_scrypt} {common_params}'

        return command_line


class AsyncOpenVINOProcess(OpenVINOPythonAPIProcess):
    def __init__(self, test, executor, log):
        super().__init__(test, executor, log)

    @staticmethod
    def __add_nstreams_for_cmd_line(command_line, nstreams):
        return f'{command_line} -nstreams {nstreams}'

    @staticmethod
    def __add_requests_for_cmd_line(command_line, requests):
        return f'{command_line} --requests {requests}'

    def get_performance_metrics(self):
        if self._status != 0 or len(self._output) == 0:
            return None, None, None

        result = self._output[-1].strip().split(',')
        average_time = float(result[0])
        fps = float(result[1])

        return average_time, fps, 0

    def _fill_command_line(self):
        path_to_async_scrypt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'inference',
                                            'inference_async_mode.py')
        python = ProcessHandler._get_cmd_python_version()

        common_params = super()._fill_command_line()
        command_line = f'{python} {path_to_async_scrypt} {common_params}'
        nstreams = self._test.dep_parameters.nstreams
        if nstreams:
            command_line = AsyncOpenVINOProcess.__add_nstreams_for_cmd_line(command_line, nstreams)
        requests = self._test.dep_parameters.async_request
        if requests:
            command_line = AsyncOpenVINOProcess.__add_requests_for_cmd_line(command_line, requests)

        return command_line


class OpenVINOParametersParser(DependentParametersParser):
    def parse_parameters(self, curr_test):
        CONFIG_FRAMEWORK_DEPENDENT_TAG = 'FrameworkDependent'
        CONFIG_FRAMEWORK_DEPENDENT_MODE_TAG = 'Mode'
        CONFIG_FRAMEWORK_DEPENDENT_EXTENSION_TAG = 'Extension'
        CONFIG_FRAMEWORK_DEPENDENT_ASYNC_REQUEST_COUNT_TAG = 'AsyncRequestCount'
        CONFIG_FRAMEWORK_DEPENDENT_THREAD_COUNT_TAG = 'ThreadCount'
        CONFIG_FRAMEWORK_DEPENDENT_STREAM_COUNT_TAG = 'StreamCount'

        dep_parameters_tag = curr_test.getElementsByTagName(CONFIG_FRAMEWORK_DEPENDENT_TAG)[0]

        _mode = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_MODE_TAG)[0].firstChild
        _extension = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_EXTENSION_TAG)[0].firstChild
        _async_request_count = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_ASYNC_REQUEST_COUNT_TAG)[0].firstChild
        _thread_count = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_THREAD_COUNT_TAG)[0].firstChild
        _stream_count = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_STREAM_COUNT_TAG)[0].firstChild

        return OpenVINOParameters(
            mode=_mode.data if _mode else None,
            extension=_extension.data if _extension else None,
            async_request_count=_async_request_count.data if _async_request_count else None,
            thread_count=_thread_count.data if _thread_count else None,
            stream_count=_stream_count.data if _stream_count else None,
        )


class OpenVINOParameters(ParametersMethods):
    def __init__(self, mode, extension, async_request_count, thread_count, stream_count):
        self.mode = None
        self.extension = None
        self.async_request = None
        self.nthreads = None
        self.nstreams = None

        if self._mode_is_correct(mode):
            self.mode = mode.title()
        if self._extension_path_is_correct(extension):
            self.extension = extension
        else:
            raise ValueError('Wrong extension path for device. File not found.')
        if self.mode == 'Sync':
            if self._parameter_not_is_none(thread_count):
                if self._int_value_is_correct(thread_count):
                    self.nthreads = int(thread_count)
                else:
                    raise ValueError('Thread count can only take values: integer greater than zero.')
        if self.mode == 'Async':
            if self._parameter_not_is_none(async_request_count):
                if self._int_value_is_correct(async_request_count):
                    self.async_request = async_request_count
                else:
                    raise ValueError('Async requiest count can only take values: integer greater than zero.')
            if self._parameter_not_is_none(stream_count):
                if self._int_value_is_correct(stream_count):
                    self.nstreams = stream_count
                else:
                    raise ValueError('Stream count can only take values: integer greater than zero.')

    @staticmethod
    def _mode_is_correct(mode):
        const_correct_mode = ['sync', 'async', 'ovbenchmark_python_latency', 'ovbenchmark_python_throughput',
                              'ovbenchmark_cpp_latency', 'ovbenchmark_cpp_throughput']
        if mode.lower() in const_correct_mode:
            return True
        raise ValueError(f'Mode is a required parameter. Mode can only take values: {", ".join(const_correct_mode)}')

    def _extension_path_is_correct(self, extension):
        if not self._parameter_not_is_none(extension) or os.path.exists(extension):
            return True
        return False


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
