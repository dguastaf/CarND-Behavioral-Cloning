import csv


class DataSource:
    def __init__(self, directory, csv_filename):
        self.directory = directory
        self.csv_filename = csv_filename
        self.csvfile = None

    def openFile(self):
        if self.csvfile:
            self.csvfile.close()

        path = '{}/{}'.format(self.directory, self.csv_filename)
        self.csvfile = open(path, 'r')
        next(self.csvfile, None)
        self.reader = csv.reader(self.csvfile)

    def nextRow(self):
        if not self.csvfile:
            self.openFile()

        try:
            return next(self.reader)
        except StopIteration:
            self.openFile()
            return next(self.reader)
