import os

from config_parser import DependentParametersParser, FrameworkParameters, Test
from framework_wrapper import FrameworkWrapper
from processes import ProcessHandler


class TensorFlowWrapper(FrameworkWrapper):
    framework_name = 'TensorFlow'

    @staticmethod
    def create_process(test, executor, log, cpp_benchmark_path=None):
        return TensorFlowProcess.create_process(test, executor, log)

    @staticmethod
    def create_test(model, dataset, indep_parameters, dep_parameters):
        return TensorFlowTest(model, dataset, indep_parameters, dep_parameters)

    @staticmethod
    def get_dependent_parameters_parser():
        return TensorFlowParametersParser()


class TensorFlowProcess(ProcessHandler):
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
    def __add_input_shape_for_cmd_line(command_line, input_shape):
        return '{0} --input_shape {1}'.format(command_line, input_shape)

    @staticmethod
    def __add_input_name_for_cmd_line(command_line, input_name):
        return '{0} --input_name {1}'.format(command_line, input_name)

    @staticmethod
    def __add_output_names_for_cmd_line(command_line, output_names):
        return '{0} --output_names {1}'.format(command_line, output_names)

    @staticmethod
    def __add_nthreads_for_cmd_line(command_line, nthreads):
        return 'OMP_NUM_THREADS={1} {0}'.format(command_line, nthreads)

    @staticmethod
    def __add_num_inter_threads_for_cmd_line(command_line, num_inter_threads):
        return '{0} --num_inter_threads {1}'.format(command_line, num_inter_threads)

    @staticmethod
    def __add_num_intra_threads_for_cmd_line(command_line, num_intra_threads):
        return '{0} --num_intra_threads {1}'.format(command_line, num_intra_threads)

    @staticmethod
    def __add_kmp_affinity_for_cmd_line(command_line, kmp_affinity):
        return 'KMP_AFFINITY={1} {0}'.format(command_line, kmp_affinity)

    @staticmethod
    def __add_raw_output_time_for_cmd_line(command_line, raw_output):
        return '{0} {1}'.format(command_line, raw_output)

    @staticmethod
    def create_process(test, executor, log):
        return TensorFlowProcess(test, executor, log)

    def get_performance_metrics(self):
        if self._status != 0 or len(self._output) == 0:
            return None, None, None

        result = self._output[-1].strip().split(',')
        average_time = float(result[0])
        fps = float(result[1])
        latency = float(result[2])

        return average_time, fps, latency

    def _fill_command_line(self):
        path_to_tensorflow_scrypt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                 'inference',
                                                 'inference_tensorflow.py')
        python = ProcessHandler._get_cmd_python_version()

        model = self._test.model.model
        dataset = self._test.dataset.path
        batch = self._test.indep_parameters.batch_size
        device = self._test.indep_parameters.device
        iteration = self._test.indep_parameters.iteration

        common_params = '-m {0} -i {1} -b {2} -d {3} -ni {4}'.format(model, dataset, batch, device, iteration)

        channel_swap = self._test.dep_parameters.channel_swap
        if channel_swap:
            common_params = TensorFlowProcess.__add_channel_swap_for_cmd_line(common_params, channel_swap)
        mean = self._test.dep_parameters.mean
        if mean:
            common_params = TensorFlowProcess.__add_mean_for_cmd_line(common_params, mean)
        input_scale = self._test.dep_parameters.input_scale
        if input_scale:
            common_params = TensorFlowProcess.__add_input_scale_for_cmd_line(common_params, input_scale)
        input_shape = self._test.dep_parameters.input_shape
        if input_shape:
            common_params = TensorFlowProcess.__add_input_shape_for_cmd_line(common_params, input_shape)
        input_name = self._test.dep_parameters.input_name
        if input_name:
            common_params = TensorFlowProcess.__add_input_name_for_cmd_line(common_params, input_name)
        output_names = self._test.dep_parameters.output_names
        if output_names:
            common_params = TensorFlowProcess.__add_output_names_for_cmd_line(common_params, output_names)
        num_inter_threads = self._test.dep_parameters.num_inter_threads
        if num_inter_threads:
            common_params = TensorFlowProcess.__add_num_inter_threads_for_cmd_line(common_params, num_inter_threads)
        num_intra_threads = self._test.dep_parameters.num_intra_threads
        if num_intra_threads:
            common_params = TensorFlowProcess.__add_num_intra_threads_for_cmd_line(common_params, num_intra_threads)

        common_params = TensorFlowProcess.__add_raw_output_time_for_cmd_line(common_params, '--raw_output true')

        command_line = '{0} {1} {2}'.format(python, path_to_tensorflow_scrypt, common_params)

        nthreads = self._test.dep_parameters.nthreads
        if nthreads:
            command_line = TensorFlowProcess.__add_nthreads_for_cmd_line(command_line, nthreads)
        kmp_affinity = self._test.dep_parameters.kmp_affinity
        if kmp_affinity:
            command_line = TensorFlowProcess.__add_kmp_affinity_for_cmd_line(command_line, kmp_affinity)

        return command_line


class TensorFlowParametersParser(DependentParametersParser):
    def parse_parameters(self, curr_test):
        CONFIG_FRAMEWORK_DEPENDENT_TAG = 'FrameworkDependent'
        CONFIG_FRAMEWORK_DEPENDENT_CHANNEL_SWAP_TAG = 'ChannelSwap'
        CONFIG_FRAMEWORK_DEPENDENT_MEAN_TAG = 'Mean'
        CONFIG_FRAMEWORK_DEPENDENT_INPUT_SCALE_TAG = 'InputScale'
        CONFIG_FRAMEWORK_DEPENDENT_INPUT_SHAPE_TAG = 'InputShape'
        CONFIG_FRAMEWORK_DEPENDENT_INPUT_NAME_TAG = 'InputName'
        CONFIG_FRAMEWORK_DEPENDENT_OUTPUT_NAMES_TAG = 'OutputNames'
        CONFIG_FRAMEWORK_DEPENDENT_THREAD_COUNT_TAG = 'ThreadCount'
        CONFIG_FRAMEWORK_DEPENDENT_INTER_OP_PARALLELISM_THREADS_TAG = 'InterOpParallelismThreads'
        CONFIG_FRAMEWORK_DEPENDENT_INTRA_OP_PARALLELISM_THREADS_TAG = 'IntraOpParallelismThreads'
        CONFIG_FRAMEWORK_DEPENDENT_KMP_AFFINITY_TAG = 'KmpAffinity'

        dep_parameters_tag = curr_test.getElementsByTagName(CONFIG_FRAMEWORK_DEPENDENT_TAG)[0]

        _channel_swap = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_CHANNEL_SWAP_TAG)[0].firstChild
        _mean = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_MEAN_TAG)[0].firstChild
        _input_scale = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_INPUT_SCALE_TAG)[0].firstChild
        _input_shape = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_INPUT_SHAPE_TAG)[0].firstChild
        _input_name = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_INPUT_NAME_TAG)[0].firstChild
        _output_names = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_OUTPUT_NAMES_TAG)[0].firstChild
        _thread_count = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_THREAD_COUNT_TAG)[0].firstChild
        _inter_op_parallelism_threads = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_INTER_OP_PARALLELISM_THREADS_TAG)[0].firstChild
        _intra_op_parallelism_threads = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_INTRA_OP_PARALLELISM_THREADS_TAG)[0].firstChild
        _kmp_affinity = dep_parameters_tag.getElementsByTagName(
            CONFIG_FRAMEWORK_DEPENDENT_KMP_AFFINITY_TAG)[0].firstChild

        return TensorFlowParameters(
            channel_swap=_channel_swap.data if _channel_swap else None,
            mean=_mean.data if _mean else None,
            input_scale=_input_scale.data if _input_scale else None,
            input_shape=_input_shape.data if _input_shape else None,
            input_name=_input_name.data if _input_name else None,
            output_names=_output_names.data if _output_names else None,
            thread_count=_thread_count.data if _thread_count else None,
            inter_op_parallelism_threads=_inter_op_parallelism_threads.data if _inter_op_parallelism_threads else None,
            intra_op_parallelism_threads=_intra_op_parallelism_threads.data if _intra_op_parallelism_threads else None,
            kmp_affinity=_kmp_affinity.data if _kmp_affinity else None,
        )


class TensorFlowParameters(FrameworkParameters):
    def __init__(self, channel_swap, mean, input_scale, input_shape, input_name, output_names, thread_count,
                 inter_op_parallelism_threads, intra_op_parallelism_threads, kmp_affinity):
        self.channel_swap = None
        self.mean = None
        self.input_scale = None
        self.input_shape = None
        self.input_name = None
        self.output_names = None
        self.nthreads = None
        self.num_inter_threads = None
        self.num_intra_threads = None
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
        if self._parameter_not_is_none(input_shape):
            if self._input_shape_is_correct(input_shape):
                self.input_shape = input_shape
            else:
                raise ValueError('Input shape can only take values: list of 3 integer elements greater than zero.')
        if self._parameter_not_is_none(input_name):
            self.input_name = input_name
        if self._parameter_not_is_none(output_names):
            self.output_names = output_names
        if self._parameter_not_is_none(thread_count):
            if self._int_value_is_correct(thread_count):
                self.nthreads = thread_count
            else:
                raise ValueError('Threads count can only take integer value')
        if self._parameter_not_is_none(inter_op_parallelism_threads):
            if self._int_value_is_correct(inter_op_parallelism_threads):
                self.num_inter_threads = inter_op_parallelism_threads
            else:
                raise ValueError('Inter op parallelism threads can only take integer value')
        if self._parameter_not_is_none(intra_op_parallelism_threads):
            if self._int_value_is_correct(intra_op_parallelism_threads):
                self.num_intra_threads = intra_op_parallelism_threads
            else:
                raise ValueError('Intra op parallelism threads can only take integer value')
        if self._parameter_not_is_none(kmp_affinity):
            self.kmp_affinity = kmp_affinity

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

    def _input_shape_is_correct(self, input_shape):
        shape_check = input_shape.split()
        if len(shape_check) != 3:
            return False
        for i in shape_check:
            if not self._int_value_is_correct(i):
                return False
        return True


class TensorFlowTest(Test):
    def __init__(self, model, dataset, indep_parameters, dep_parameters):
        super().__init__(model, dataset, indep_parameters, dep_parameters)

    def get_report(self):
        report_res = ('{0};{1};{2};{3};{4};input_shape;{5};{6};Sync;Device: {7}, Iteration count: {8}, '
                      'Thread count: {9}, Inter threads: {10}, Intra threads: {11}, KMP_AFFINITY: {12}').format(
            self.model.task, self.model.name, self.dataset.name, self.model.source_framework,
            self.indep_parameters.inference_framework, self.model.precision,
            self.indep_parameters.batch_size, self.indep_parameters.device,
            self.indep_parameters.iteration, self.dep_parameters.nthreads,
            self.dep_parameters.num_inter_threads, self.dep_parameters.num_intra_threads,
            self.dep_parameters.kmp_affinity)

        return report_res
