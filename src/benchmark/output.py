import pandas as pd

from pathlib import Path


class OutputHandler:
    def __init__(self, table_name):
        self.__table_name = table_name

    def get_table_name(self):
        return self.__table_name

    @staticmethod
    def __create_table_row(executor, test, process):
        status = 'Success' if process.get_status() == 0 else 'Failed'
        test_parameters = test.get_report().replace('input_shape', process.get_model_shape())
        average_time, fps, latency = process.get_performance_metrics()
        hardware_info = executor.get_infrastructure()

        return '{0};{1};{2};{3};{4};{5}'.format(status, test_parameters, hardware_info, average_time, latency, fps)

    def create_table(self, custom_table=None, custom_headers=None):
        HEADERS = custom_headers or 'Status;Task type;Topology name;Dataset;Framework;Inference Framework;Input blob sizes;Precision;Batch size;Mode;Parameters;Infrastructure;Average time of single pass (s);Latency;FPS'  # noqa: E501
        table_name = custom_table or self.__table_name
        if not Path(table_name).exists():
            with open(table_name, 'w') as table:
                table.write(HEADERS + '\n')
                table.close()

    def add_row_to_table(self, **kwargs):  # executor, test, process || filename, row
        if len(kwargs) == 2:
            filename = kwargs['filename']
            report_row = kwargs['row']
        elif len(kwargs) == 3:
            filename = self.__table_name
            report_row = self.__create_table_row(**kwargs)
        else:
            raise ValueError('add_row_to_table receives only kwargs: executor,test,process || filename,row')
        with open(filename, 'a') as table:
            table.write(report_row + '\n')
            table.close()
