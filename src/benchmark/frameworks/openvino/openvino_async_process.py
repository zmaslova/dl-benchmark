import os

from .openvino_python_api_process import OpenVINOPythonAPIProcess
from ..processes import ProcessHandler


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
