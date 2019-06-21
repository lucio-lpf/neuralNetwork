import numpy.matlib as np
import math
from random import randint
from copy import copy, deepcopy
import csv

class NeuralNetwork:

    def __init__(self, entradas, camadas, initial_weights_file, fator_regularizacao):

        print("Inicializando matriz de pesos da rede neural")
        self.fator_regularizacao = fator_regularizacao
        self.pesos_matriz = [[[] for x in range(camadas[y])] for y in range(len(camadas))]
        self.bias_matriz = [[[] for x in range(camadas[y])] for y in range(len(camadas))]
        if initial_weights_file is None:
            for index, camada in enumerate(camadas):
                for index_j in range(0, camadas[index]):
                    if index is 0:
                        self.pesos_matriz[index][index_j] = [randint(1,9) for _ in range(0, entradas)]
                    else:
                        self.pesos_matriz[index][index_j] = [randint(1,9) for _ in range(0, camadas[index - 1])]
                    self.bias_matriz[index][index_j] = randint(1, 9) / 10
        else:
            with open(initial_weights_file) as weights_file:
                dataFrame = csv.reader(weights_file, delimiter=",", quoting=csv.QUOTE_NONE)
                for index, file_row in enumerate(dataFrame):
                    passou_bias = False
                    neuronio_num = 0
                    for weight in file_row:
                        if passou_bias is False:
                            self.bias_matriz[index][neuronio_num] = float(weight)
                            passou_bias = True
                        elif ";" in weight:
                            ultimo_peso, prox_bias = weight.split(";")
                            self.pesos_matriz[index][neuronio_num].append(float(ultimo_peso))
                            neuronio_num += 1
                            self.bias_matriz[index][neuronio_num] = float(prox_bias)
                        else:
                            self.pesos_matriz[index][neuronio_num].append(float(weight))
        self.print_matrizes()

    def print_matrizes(self):
        print("Bias por neuronio:")
        for index, line in enumerate(self.bias_matriz):
            print("Camada: ", index, "  ", line)
        print("Pesos das camadas:")
        for index, line in enumerate(self.pesos_matriz):
            print("Camada: ", index, "  ", line)

    def treina_rede(self, atributos, resultado, alfa, dataset, results):
        pesos_mat = deepcopy(self.pesos_matriz)
        ativacao_matriz = self.calcula_saidas(atributos)
        saida_da_rede = ativacao_matriz[len(ativacao_matriz) - 1]

        delta_matriz = self.calcula_deltas(ativacao_matriz, resultado)

        gradientes_matriz = self.calcula_gradientes(atributos, ativacao_matriz, delta_matriz)
        gradientes_matriz_bias = deepcopy(delta_matriz)

        custo = self.calcula_custos(dataset, results)

        self.atualiza_pesos(gradientes_matriz, 0.01, custo)

        custo = self.calcula_custos(dataset, results)

        return custo

    def calcula_saidas(self, registro):
        matriz_de_saidas = [[0 for x in range(len(self.pesos_matriz[y]))] for y in range(len(self.pesos_matriz))]
        for index in range(0, len(matriz_de_saidas)):
            if index is 0:
                for index_j in range(0, len(matriz_de_saidas[index])):
                    matriz_de_saidas[0][index_j] = self.sigmoide(np.matmul(registro, self.pesos_matriz[index][index_j]) + self.bias_matriz[index][index_j])
            else:
                for index_j in range(0, len(matriz_de_saidas[index])):
                    matriz_de_saidas[index][index_j] = self.sigmoide(np.matmul(matriz_de_saidas[index - 1], self.pesos_matriz[index][index_j]) + self.bias_matriz[index][index_j])

        return matriz_de_saidas

    def calcula_deltas(self, saidas_matriz, resultados_corretos):
        delta_matriz = [[0 for x in range(len(saidas_matriz[y]))] for y in range(len(saidas_matriz))]
        index_camada_saidas = len(saidas_matriz) - 1

        for index_delta in range(len(saidas_matriz[index_camada_saidas])):
            delta_matriz[index_camada_saidas][index_delta] = saidas_matriz[index_camada_saidas][index_delta] - resultados_corretos[index_delta]

        for index_camada in reversed(range(len(delta_matriz) - 1)):
            for index_delta in range(len(delta_matriz[index_camada])):
                delta_matriz[index_camada][index_delta] = saidas_matriz[index_camada][index_delta]*(1 - saidas_matriz[index_camada][index_delta])
                soma_pesos = 0
                for index, peso in enumerate(self.pesos_matriz[index_camada + 1]):
                    soma_pesos += peso[index_delta]*delta_matriz[index_camada + 1][index]
                delta_matriz[index_camada][index_delta] = delta_matriz[index_camada][index_delta]*soma_pesos

        return delta_matriz

    def calcula_gradientes(self, registro, ativacao_matriz, delta_matriz):
        gradiente_matriz = [[[] for x in range(len(self.pesos_matriz[y]))] for y in range(len(self.pesos_matriz))]
        for index_camada in range(len(gradiente_matriz)):
            for index_neuron in range(len(gradiente_matriz[index_camada])):
                for index_gradiente in range(len(self.pesos_matriz[index_camada][index_neuron])):
                    if index_camada is 0:
                        valor_gradiente_peso = registro[index_gradiente] * delta_matriz[index_camada][index_neuron] + self.fator_regularizacao*self.pesos_matriz[index_camada][index_neuron][index_gradiente]
                        gradiente_matriz[index_camada][index_neuron].append(valor_gradiente_peso)
                    else:
                        valor_gradiente_peso = ativacao_matriz[index_camada - 1][index_gradiente]*delta_matriz[index_camada][index_neuron] + self.fator_regularizacao*self.pesos_matriz[index_camada][index_neuron][index_gradiente]
                        gradiente_matriz[index_camada][index_neuron].append(valor_gradiente_peso)
        return gradiente_matriz

    def calcula_custos(self, dataset, resultados):
        custo_total = 0
        for index_dado, dado in enumerate(dataset):
            saidas_calculada = self.calcula_saidas(dado)[-1]
            for index_saida, saida in enumerate(saidas_calculada):
                parte_ativa = -resultados[index_dado][index_saida] * (math.log10(saida))
                parte_inativa = - (1 - resultados[index_dado][index_saida])*(math.log10(1 - saida))
                custo_total += parte_ativa + parte_inativa

        custo_total = custo_total/len(dataset) + self.calcula_taxa_regularizacao(len(dataset))
        return custo_total


    def calcula_taxa_regularizacao(self, tamanho_dados):
        soma_thetas = 0
        for camadas in self.pesos_matriz:
            for neuronio in camadas:
                for peso in neuronio:
                    soma_thetas += pow(peso, 2)
        return ((soma_thetas* self.fator_regularizacao)/(2*tamanho_dados))


    def sigmoide(self, funcao):
        sig = 1 / (1 + math.exp(-funcao))
        return sig

    def verificacao_numerica(self):
        pass

    def atualiza_pesos(self, gradientes_matriz, alpha, custo):
        for index_camada in range(len(self.pesos_matriz)):
            for index_neuronio in range(len(self.pesos_matriz[index_camada])):
                for index_peso in range(len(self.pesos_matriz[index_camada][index_neuronio])):
                    self.pesos_matriz[index_camada][index_neuronio][index_peso] -= alpha*gradientes_matriz[index_camada][index_neuronio][index_peso]*custo
