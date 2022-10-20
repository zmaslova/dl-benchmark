import os

from config_parser import DependentParametersParser, FrameworkParameters, Test
from framework_wrapper import FrameworkWrapper
from processes import ProcessHandler


class IntelCaffeWrapper(FrameworkWrapper):
    framework_name = 'Caffe'

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        return IntelCaffeProcess.create_process(test, executor, log)

    @staticmethod
    def create_test(model, dataset, indep_parameters, dep_parameters):
        return IntelCaffeTest(model, dataset, indep_parameters, dep_parameters)

    @staticmethod
    def get_dependent_parameters_parser():
        return IntelCaffeParametersParser()


class IntelCaffeProcess(ProcessHandler):
    def __init__(self, test, executor, log):
        super().__init__(test, executor, log)

    @staticmethod
    def __add_channel_swap_for_cmd_line(command_line, channel_swap):
        return '{0} --channel_swap {1}'.format(command_line, channel_swap)

    @staticmethod
    def __add_mean_for_cmd_line(command_line, mean):
        return '{0} --mean {1}'.format(command_line, mean)

    @staticmethod
    def __add_input_scale_for_cmd_line(command_line, input_scale):
        return '{0} --input_scale {1}'.format(command_line, input_scale)

    @staticmethod
    def __add_nthreads_for_cmd_line(command_line, nthreads):
        return 'OMP_NUM_THREADS={1} {0}'.format(command_line, nthreads)

    @staticmethod
    def __add_kmp_affinity_for_cmd_line(command_line, kmp_affinity):
        return 'KMP_AFFINITY={1} {0}'.format(command_line, kmp_affinity)

    @staticmethod
    def __add_raw_output_time_for_cmd_line(command_line, raw_output):
        return '{0} {1}'.format(command_line, raw_output)

    @staticmethod
    def create_process(test, executor, log):
        return IntelCaffeProcess(test, executor, log)

    def get_performance_metrics(self):
        if self._status != 0 or len(self._output) == 0:
            return None, None, None

        result = self._output[-1].strip().split(',')
        average_time = float(result[0])
        fps = float(result[1])
        latency = float(result[2])

        return average_time, fps, latency

    def _fill_command_line(self):
        path_to_intelcaffe_scrypt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                 'inference',
                                                 'inference_caffe.py')
        python = ProcessHandler._get_cmd_python_version()

        model_prototxt = self._test.model.model
        model_caffemodel = self._test.model.weight
        dataset = self._test.dataset.path
        batch = self._test.indep_parameters.batch_size
        device = self._test.indep_parameters.device
        iteration = self._test.indep_parameters.iteration

        common_params = '-m {0} -w {1} -i {2} -b {3} -d {4} -ni {5}'.format(
            model_prototxt, model_caffemodel, dataset, batch, device, iteration)
        channel_swap = self._test.dep_parameters.channel_swap
        if channel_swap:
            common_params = IntelCaffeProcess.__add_channel_swap_for_cmd_line(common_params, channel_swap)
        mean = self._test.dep_parameters.mean
        if mean:
            common_params = IntelCaffeProcess.__add_mean_for_cmd_line(common_params, mean)
        input_scale = self._test.dep_parameters.input_scale
        if input_scale:
            common_params = IntelCaffeProcess.__add_input_scale_for_cmd_line(common_params, input_scale)

        common_params = IntelCaffeProcess.__add_raw_output_time_for_cmd_line(common_params, '--raw_output true')
        command_line = '{0} {1} {2}'.format(python, path_to_intelcaffe_scrypt, common_params)

        nthreads = self._test.dep_parameters.nthreads
        if nthreads:
            command_line = IntelCaffeProcess.__add_nthreads_for_cmd_line(command_line, nthreads)
        kmp_affinity = self._test.dep_parameters.kmp_affinity
        if kmp_affinity:
            command_line = IntelCaffeProcess.__add_kmp_affinity_for_cmd_line(command_line, kmp_affinity)

        return command_line


class IntelCaffeParametersParser(DependentParametersParser):
    def parse_parameters(self, curr_test):
        CONFIG_FRAMEWORK_DEPENDENT_TAG = 'FrameworkDependent'
        CONFIG_FRAMEWORK_DEPENDENT_CHANNEL_SWAP_TAG = 'ChannelSwap'
        CONFIG_FRAMEWORK_DEPENDENT_MEAN_TAG = 'Mean'
        CONFIG_FRAMEWORK_DEPENDENT_INPUT_SCALE_TAG = 'InputScale'
        CONFIG_FRAMEWORK_DEPENDENT_THREAD_COUNT_TAG = 'ThreadCount'
        CONFIG_FRAMEWORK_DEPENDENT_KMP_AFFINITY_TAG = 'KmpAffinity'

        dep_parameters_tag = curr_test.getElementsByTagName(CONFIG_FRAMEWORK_DEPENDENT_TAG)[0]

        _channel_swap = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_CHANNEL_SWAP_TAG)[0].firstChild
        _mean = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_MEAN_TAG)[0].firstChild
        _input_scale = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_INPUT_SCALE_TAG)[0].firstChild
        _thread_count = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_THREAD_COUNT_TAG)[0].firstChild
        _kmp_affinity = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_KMP_AFFINITY_TAG)[0].firstChild

        return IntelCaffeParameters(
            channel_swap=_channel_swap.data if _channel_swap else None,
            mean=_mean.data if _mean else None,
            input_scale=_input_scale.data if _input_scale else None,
            thread_count=_thread_count.data if _thread_count else None,
            kmp_affinity=_kmp_affinity.data if _kmp_affinity else None,
        )


class IntelCaffeParameters(FrameworkParameters):
    def __init__(self, channel_swap, mean, input_scale, thread_count, kmp_affinity):
        self.channel_swap = None
        self.mean = None
        self.input_scale = None
        self.nthreads = None
        self.kmp_affinity = None

        if self._parameter_not_is_none(channel_swap):
            if self._channel_swap_is_correct(channel_swap):
                self.channel_swap = channel_swap
            else:
                raise ValueError('Channel swap can only take values: list of unique values 0, 1, 2.')
        if self._parameter_not_is_none(mean):
            if self._mean_is_correct(mean):
                self.mean = mean
            else:
                raise ValueError('Mean can only take values: list of 3 float elements.')
        if self._parameter_not_is_none(input_scale):
            if self._float_value_is_correct(input_scale):
                self.input_scale = input_scale
            else:
                raise ValueError('Input scale can only take values: float greater than zero.')
        if self._parameter_not_is_none(thread_count):
            if self._int_value_is_correct(thread_count):
                self.nthreads = thread_count
            else:
                raise ValueError('Threads count can only take integer value')
        if self._parameter_not_is_none(kmp_affinity):
            self.kmp_affinity = kmp_affinity

    @staticmethod
    def _channel_swap_is_correct(channel_swap):
        set_check = {'0', '1', '2'}
        set_in = set(channel_swap.split())
        return set_in == set_check

    def _mean_is_correct(self, mean):
        mean_check = mean.split()
        if len(mean_check) != 3:
            return False
        for i in mean_check:
            if not self._float_value_is_correct(i):
                return False
        return True


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
