from .openvino_parameters import OpenVINOParameters
from ..config_parser.dependent_parameters_parser import DependentParametersParser


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
