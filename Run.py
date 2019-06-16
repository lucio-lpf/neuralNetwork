from DataHandler import DataHandler
from NeuralNetwork import *
import sys
import csv

alpha = 0.01

#treina o modelo

def main():

    network_file = "network_default.txt"
    args = sys.argv[1:]
    if len(args) is 3:
        network_file = args[0]
        initial_weights_file = args[1]
        dataset_file = args[2]

    else:
        print("Selecione seu dataset: \n 1 - Ionosphere \n 2 - Pima \n 3 - Wdbc \n 4 - Wine \n 5 - Teste")
        escolha = int(input("Escolha: "))
        if escolha is 1:
            dataset_file = "./datasets/ionosphere.csv"
        elif escolha is 2:
            dataset_file = "./datasets/pima.csv"
        elif escolha is 3:
            dataset_file = "./datasets/wdbc.csv"
        elif escolha is 4:
            dataset_file = "./datasets/wine.csv"
        elif escolha is 5:
            dataset_file = "./datasets/teste.csv"
        else:
            print("Escolha invalida")
            exit()

    camadas = []
    fator_regularizacao = None
    with open(network_file) as network:
        info_camadas = csv.reader(network, delimiter=",", quoting=csv.QUOTE_NONE)
        for index, row in enumerate(info_camadas):
            if index is 0:
                fator_regularizacao = float(row[0])
            else:
                camadas.append(int(row[0]))

    dataset = DataHandler(dataset_file)
    dataset.normalizeData()

    entradas = len(dataset.data[0])
    nn = NeuralNetwork(entradas, camadas)
    for data in dataset.data:
        nn.treina_rede(dataset.data[0], dataset.results[0], alpha)

if __name__ == '__main__':
    main()
