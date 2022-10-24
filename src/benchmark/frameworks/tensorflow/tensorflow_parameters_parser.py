from .tensorflow_parameters import TensorFlowParameters
from ..config_parser.dependent_parameters_parser import DependentParametersParser


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
