from benchmark.config_parser import FrameworkParameters


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
