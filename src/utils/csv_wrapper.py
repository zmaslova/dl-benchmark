import csv


class CsvReader:
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, 'r') as csv_file:
            dialect = csv.Sniffer().sniff(csv_file.read(1024))
            csv_file.seek(0)
            reader = csv.DictReader(csv_file, dialect=dialect)
            return [row for row in reader]

class CsvWriter:
    def __init__(self, path, headers, delimiter = ';'):
        self.path = path
        self.headers = headers
        self.delimiter = delimiter     
    
    def write_headers(self):
        with open(self.path, 'w') as csv_file:
            self._writer = csv.DictWriter(csv_file, fieldnames=self.headers, dialect=csv.excel, 
                                          delimiter=self.delimiter, quoting=csv.QUOTE_ALL)
            self._writer.writeheader()

    def append_row(self, row_dict):
        with open(self.path, 'a') as csv_file:
            self._writer = csv.DictWriter(csv_file, fieldnames=self.headers, dialect=csv.excel, 
                                          delimiter=self.delimiter, quoting=csv.QUOTE_ALL)
            self._writer.writerow(row_dict)


