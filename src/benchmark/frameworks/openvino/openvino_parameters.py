import os

from benchmark.config_parser import FrameworkParameters


class OpenVINOParameters(FrameworkParameters):
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
