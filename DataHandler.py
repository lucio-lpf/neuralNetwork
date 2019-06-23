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
            # dataset = []
            # for row in dataFrame:
            #     dataset.append(row)
            # random.shuffle(dataset)
            # print(dataset)
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
        zip_content = list(zip(all_data,results))
        random.shuffle(zip_content)
        all_data, results = zip(*zip_content)
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

    def generate_batches(self, batch_number):
        tamanho_dataset = len(self.data)
        tamanho_novos_dataset = tamanho_dataset/batch_number
        array_batches = []
        array_batches_results = []
        index_dados = 0
        while len(array_batches) is not batch_number:
            batch_dados = []
            batch_resultados = []
            while len(batch_dados) < tamanho_novos_dataset:
                try:
                    batch_dados.append(self.data[index_dados])
                    batch_resultados.append(self.results[index_dados])
                except:
                    break
                index_dados += 1
            array_batches.append(batch_dados)
            array_batches_results.append(batch_resultados)
        return array_batches, array_batches_results

    def ajustar_novo_dataset(self):
        file = "Datasets/galaxy.csv"
        with open(file) as f:
            dataFrame = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
            data = []
            for row in dataFrame:
                data.append(row[3:])
            data.pop(0)
            a = open("galaxy.csv", "w")
            write = csv.writer(a, delimiter=",", quoting=csv.QUOTE_NONE)
            for row in data:
                ultimo = row.pop()
                penultimo = row.pop()
                antepenultimo = row.pop()
                dado = row.pop()
                dado += ';'
                dado += antepenultimo
                row.append(dado)
                row.append(penultimo)
                row.append(ultimo)
                write.writerow(row)