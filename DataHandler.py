import math
import csv
import random


class DataHandler:
    def __init__(self, file):
        self.data, self.results = self.openDataframe(file)

    def openDataframe(self, file):
        with open(file) as f:
            dataFrame = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
            all_data = []
            results = []
            for row in dataFrame:
                data = []
                result = []
                i = 0
                while ';' not in row[i]:
                    data.append(float(row[i]))
                    i += 1
                last_atribute, first_output = row[i].split(';')
                data.append(float(last_atribute))
                result.append(float(first_output))
                i += 1
                while i is not len(row):
                    result.append(float(row[i]))
                    i += 1
                all_data.append(data)
                results.append(result)
        return all_data, results

    def normalizeData(self):
        max = [None] * (len(self.data[0]))
        min = [None] * (len(self.data[0]))
        for row in self.data:
            for index in range(0, len(row)):
                row[index] = row[index]
                if max[index] is None or row[index] > max[index]:
                    max[index] = row[index]
                if min[index] is None or row[index] < min[index]:
                    min[index] = row[index]
        print("Normalizando dados")
        for row in self.data:
            for index in range(0, len(row)):
                row[index] = 2 * ((row[index] - min[index]) / (max[index] - min[index])) - 1