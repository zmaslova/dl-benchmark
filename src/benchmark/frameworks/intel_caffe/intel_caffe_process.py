import os

from benchmark.processes import ProcessHandler


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
