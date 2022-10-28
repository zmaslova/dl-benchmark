import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('utils')))
from csv_wrapper import CsvReport  # noqa: E402


class OutputHandler:
    def __init__(self, table_name):
        self.__table_name = table_name

        self._header_long_names = {
            'status': 'Status',
            'task': 'Task type',
            'model': 'Topology name',
            'source_framework': 'Framework',
            'launcher': 'Inference Framework',
            'device': 'Device',
            'hardware': 'Infrastructure',
            'dataset': 'Dataset',
            'metric': 'Accuracy type',
            'precision': 'Precision',
            'accuracy': 'Accuracy',
        }

        self._report = CsvReport(self.__table_name, self._header_long_names.values())

    def create_table(self):
        self._report.write_headers()

    def add_results(self, test, process, executor):
        results = process.get_result_parameters()
        hardware_info = executor.get_infrastructure()
        for _, result in enumerate(results):
            result_dict = result.get_result_dict()
            result_dict['hardware'] = hardware_info

            row_dict = {long_name: result_dict[dict_name] for dict_name, long_name in self._header_long_names.items()}
            self._report.append_row(row_dict)
