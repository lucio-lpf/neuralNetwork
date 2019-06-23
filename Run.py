from DataHandler import DataHandler
from NeuralNetwork import *
from graphs import Graphs
import sys
import csv
from random import shuffle

alpha = 4
num_batches = 3

#treina o modelo

def main():
    network_file = "network_default.txt"
    initial_weights_file = None
    args = sys.argv[1:]
    if len(args) is 3:
        network_file = args[0]
        initial_weights_file = args[1]
        dataset_file = args[2]


    else:
        print("Selecione seu dataset: \n 1 - Ionosphere \n 2 - Pima \n 3 - Wdbc \n 4 - Wine \n 5 - Teste \n 6 - Galaxy ")
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
        elif escolha is 6:
            dataset_file = "./datasets/galaxy1.csv"
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

    entradas = len(dataset.data[0])

    nn = NeuralNetwork(entradas, camadas, initial_weights_file, fator_regularizacao)
    custo = nn.calcula_custos(dataset.data, dataset.results)

    if len(args) is 3:
        print("FATOR DE REGULARIZACAO: ", fator_regularizacao)
        print("PESOS INICIAIS:")
        nn.print_matrizes()
        for index_data, data in enumerate(dataset.data):
            print("ENTRADA: ", data)

            print("SAIDAS:")
            ativacao_matriz = nn.calcula_saidas(data)
            printMatriz(ativacao_matriz)

            deltas = nn.calcula_deltas(ativacao_matriz, dataset.results[index_data])

            gradiente_bias = deltas
            gradiente_pesos = nn.calcula_gradientes(data, ativacao_matriz, deltas)
            print("GRADIENTES DOS BIAS PARA ENTRADA:")
            printMatriz(gradiente_bias)
            print("GRADIENTES DOS PESOS PARA ENTRADA:")
            printMatriz(gradiente_pesos)
            nn.atuliza_matriz_gradientes(gradiente_bias, gradiente_pesos)
        print("\n \n =================== FIM TREINAMENTO ==================== \n")
        nn.calcula_gradientes_total_regularizados(index_data + 1)
        print("GRADIENTES BIAS FINAIS DO DATASET:")
        printMatriz(nn.gradientes_bias)
        print("GRADIENTES PESOS FINAIS DO DATASET:")
        printMatriz(nn.gradientes)
        nn.atualiza_pesos(alpha)
        print("NOVOS PESSO:")
        nn.print_matrizes()
        custo = nn.calcula_custos(dataset.data, dataset.results)
        print("Custo total: ", custo)
    else:

        dataset.normalizeData()
        batches_dados, batches_resultados = dataset.generate_batches(num_batches)
        saida_da_rede = []
        i = 0
        custos = []
        epocas = 0
        while custo > 0.01:
            print(custo)
            for index_batch, batch_dados in enumerate(batches_dados):
                for index_entrada, entrada in enumerate(batch_dados):
                    nn.treina_rede(entrada, batches_resultados[index_batch][index_entrada])
                nn.calcula_gradientes_total_regularizados(index_entrada + 1)
                nn.atualiza_pesos(alpha)
                epocas += 1

                nn.gradientes = None
                nn.gradientes_bias = None


            custo = nn.calcula_custos(dataset.data, dataset.results)
            custos.append(custo)
            i = i + 1
            if i == 5:
                print(custo)
                i = 0
                saida_da_rede = []
                for index, data in enumerate(dataset.data):
                    saida_da_rede.append(nn.calcula_saidas(data)[-1])
                g = Graphs()
                g.classificacao(dataset.results, saida_da_rede, epocas, enfase_f1_score=1, custo=custo, custos=custos)
        nn.print_matrizes()
        for index, data in enumerate(dataset.data):
            saida_da_rede.append(nn.calcula_saidas(data)[-1])
        g = Graphs()
        g.classificacao(dataset.results, saida_da_rede, epocas, enfase_f1_score=1, custo=custo, custos=custos)

def createKFolds(dataFrame, k):
    shuffle(dataFrame)
    listOfDataFrames = []
    size = len(dataFrame) // k
    next = len(dataFrame) // k
    for index in range(0, k):
        listOfDataFrames.append(dataFrame[index * size:next])
        next = next + size
    return listOfDataFrames

def printMatriz(matriz):
    for index, line in enumerate(matriz):
        print("\t Camada ", index, ": ", line)
    print("\n")

if __name__ == '__main__':
    main()