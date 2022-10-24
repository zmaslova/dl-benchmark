from benchmark.frameworks.openvino.openvino_async_process import AsyncOpenVINOProcess
from benchmark.frameworks.openvino.openvino_benchmark_cpp_process import OpenVINOBenchmarkCppProcess
from benchmark.frameworks.openvino.openvino_benchmark_python_process import OpenVINOBenchmarkPythonProcess
from benchmark.frameworks.openvino.openvino_sync_process import SyncOpenVINOProcess


def create_process(test, executor, log, cpp_benchmark_path=None):
    mode = test.dep_parameters.mode.lower()
    if mode == 'sync':
        return SyncOpenVINOProcess(test, executor, log)
    elif mode == 'async':
        return AsyncOpenVINOProcess(test, executor, log)
    elif mode == 'ovbenchmark_python_latency':
        return OpenVINOBenchmarkPythonProcess(test, executor, log, 'latency')
    elif mode == 'ovbenchmark_python_throughput':
        return OpenVINOBenchmarkPythonProcess(test, executor, log, 'throughput')
    elif mode == 'ovbenchmark_cpp_latency':
        return OpenVINOBenchmarkCppProcess(test, executor, log, cpp_benchmark_path, 'latency')
    elif mode == 'ovbenchmark_cpp_throughput':
        return OpenVINOBenchmarkCppProcess(test, executor, log, cpp_benchmark_path, 'throughput')
