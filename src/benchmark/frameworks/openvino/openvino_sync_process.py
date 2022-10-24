import os

from .openvino_python_api_process import OpenVINOPythonAPIProcess
from ..processes import ProcessHandler


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
