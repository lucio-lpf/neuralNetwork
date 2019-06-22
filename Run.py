from DataHandler import DataHandler
from NeuralNetwork import *
from graphs import Graphs
import sys
import csv
from random import shuffle

alpha = 0.5

#treina o modelo

def main():

    network_file = "network_default.txt"
    initial_weights_file = None#"initial_weights.txt"
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
            dataset_file = "./datasets/teste2.csv"
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
    batches_dados, batches_resultados = dataset.generate_batches(1)
    #dataset.normalizeData()

    entradas = len(dataset.data[0])

    nn = NeuralNetwork(entradas, camadas, initial_weights_file, fator_regularizacao)
    custo = nn.calcula_custos(dataset.data, dataset.results)


    saida_da_rede = []
    i = 0
    print(custo)
    while custo > 0.4:
        print(custo)
        for index_batch, batch_dados in enumerate(batches_dados):
            for index_entrada, entrada in enumerate(batch_dados):
                nn.treina_rede(entrada, batches_resultados[index_batch][index_entrada])
            nn.calcula_gradientes_total_regularizados(index_entrada + 1)
            nn.atualiza_pesos(alpha)
            custo = nn.calcula_custos(dataset.data, dataset.results)
            nn.gradientes = None
            nn.gradientes_bias = None

            i = i + 1
            if i == 500:
                print(custo)
                i = 0
                for index, data in enumerate(dataset.data):
                    saida_da_rede.append(nn.calcula_saidas(data)[-1])
                g = Graphs()
                g.classificacao(dataset.results, saida_da_rede, enfase_f1_score=1)

    nn.print_matrizes()
    for index, data in enumerate(dataset.data):
        saida_da_rede.append(nn.calcula_saidas(data)[-1])
    g = Graphs()
    g.classificacao(dataset.results, saida_da_rede, enfase_f1_score=1)

def createKFolds(dataFrame, k):
    shuffle(dataFrame)
    listOfDataFrames = []
    size = len(dataFrame) // k
    next = len(dataFrame) // k
    for index in range(0, k):
        listOfDataFrames.append(dataFrame[index * size:next])
        next = next + size
    return listOfDataFrames

if __name__ == '__main__':
    main()
