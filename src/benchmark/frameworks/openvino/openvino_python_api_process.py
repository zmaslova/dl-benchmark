from benchmark.frameworks.openvino.openvino_process import OpenVINOProcess


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
