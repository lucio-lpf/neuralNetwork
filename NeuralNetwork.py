import numpy.matlib as np
import math
from random import randint
from copy import deepcopy
import csv
import numpy as np

class NeuralNetwork:

    def __init__(self, entradas, camadas, initial_weights_file, fator_regularizacao):

        print("Inicializando matriz de pesos da rede neural")
        self.gradientes = None
        self.gradientes_bias = None
        self.fator_regularizacao = fator_regularizacao
        self.pesos_matriz = [[[] for x in range(camadas[y])] for y in range(len(camadas))]
        self.bias_matriz = [[[] for x in range(camadas[y])] for y in range(len(camadas))]
        if initial_weights_file is None:
            for index, camada in enumerate(camadas):
                for index_j in range(0, camadas[index]):
                    if index is 0:
                        self.pesos_matriz[index][index_j] = [randint(1, 9) for _ in range(0, entradas)]
                    else:
                        self.pesos_matriz[index][index_j] = [randint(1, 9) for _ in range(0, camadas[index - 1])]
                    self.bias_matriz[index][index_j] = randint(1, 9)/ 10
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

    def treina_rede(self, atributos, resultado):

        ativacao_matriz = self.calcula_saidas(atributos)

        delta_matriz = self.calcula_deltas(ativacao_matriz, resultado)

        gradiente_da_entrada = self.calcula_gradientes(atributos, ativacao_matriz, delta_matriz)

        self.atuliza_matriz_gradientes(delta_matriz, gradiente_da_entrada)

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
                        valor_gradiente_peso = registro[index_gradiente] * delta_matriz[index_camada][index_neuron]
                        gradiente_matriz[index_camada][index_neuron].append(valor_gradiente_peso)
                    else:
                        valor_gradiente_peso = ativacao_matriz[index_camada - 1][index_gradiente]*delta_matriz[index_camada][index_neuron]
                        gradiente_matriz[index_camada][index_neuron].append(valor_gradiente_peso)
        return gradiente_matriz

    def atuliza_matriz_gradientes(self, novos_bias, nova_matriz_greadiente):
        if self.gradientes is None:
            self.gradientes = nova_matriz_greadiente
            self.gradientes_bias = novos_bias
        else:
            for index_camada, camada in enumerate(self.gradientes):
                a = np.array(nova_matriz_greadiente[index_camada]) + np.array(self.gradientes[index_camada])
                self.gradientes[index_camada] = a.tolist()
                b = np.array(novos_bias[index_camada]) + np.array(self.gradientes_bias[index_camada])
                self.gradientes_bias[index_camada] = b.tolist()

    def calcula_gradientes_total_regularizados(self, numero_entradas):
        for index_camada in range(len(self.gradientes)):
            for index_neuronio in range(len(self.gradientes[index_camada])):
                self.gradientes_bias[index_camada][index_neuronio] = self.gradientes_bias[index_camada][index_neuronio]/numero_entradas
                for index_peso in range(len(self.gradientes[index_camada][index_neuronio])):
                    self.gradientes[index_camada][index_neuronio][index_peso] += self.fator_regularizacao*self.pesos_matriz[index_camada][index_neuronio][index_peso]
                    self.gradientes[index_camada][index_neuronio][index_peso] = self.gradientes[index_camada][index_neuronio][index_peso]/numero_entradas

    def calcula_custos(self, dataset, resultados):
        custo_total = 0
        for index_dado, dado in enumerate(dataset):
            saidas_calculada = self.calcula_saidas(dado)[-1]
            for index_saida, saida in enumerate(saidas_calculada):
                parte_ativa = -resultados[index_dado][index_saida] * (math.log10(saida))
                parte_inativa = 0
                if saida is not 1:
                    parte_inativa = - (1 - resultados[index_dado][index_saida])*(math.log10(1 - saida))
                custo_total += parte_ativa + parte_inativa

        custo_total = custo_total/(2*len(dataset)) + self.calcula_taxa_regularizacao(len(dataset))
        return custo_total

    def calcula_custo_entrada(self, entrada, resultado):
        saidas_calculada = self.calcula_saidas(entrada)[-1]
        custo_total = 0
        for index_saida, saida in enumerate(saidas_calculada):
            parte_ativa = -resultado[index_saida] * (-math.log10(saida))
            parte_inativa = 0
            if saida is not 1:
                parte_inativa = - (1 - resultado[index_saida]) * (-math.log10(1 - saida))
            custo_total += parte_ativa + parte_inativa
        return custo_total

    def calcula_taxa_regularizacao(self, tamanho_dados):
        soma_thetas = 0
        for camadas in self.pesos_matriz:
            for neuronio in camadas:
                for peso in neuronio:
                    soma_thetas += pow(peso, 2)
        return ((soma_thetas* self.fator_regularizacao)/(2*tamanho_dados))


    def sigmoide(self, funcao):
        if funcao < -20:
            return 0
        if funcao > 20:
            return 1
        sig = 1 / (1 + math.exp(-funcao))
        return sig

    def verificacao_numerica(self, dataset, results, epsilum):
        gradiente_matriz = [[[] for x in range(len(self.pesos_matriz[y]))] for y in range(len(self.pesos_matriz))]
        for index_camada in range(len(self.pesos_matriz)):
                for index_neuronio in range(len(self.pesos_matriz[index_camada])):
                    for index_peso in range(len(self.pesos_matriz[index_camada][index_neuronio])):
                        self.pesos_matriz[index_camada][index_neuronio][index_peso] += epsilum
                        custo_maior = self.calcula_custos(dataset, results)
                        self.pesos_matriz[index_camada][index_neuronio][index_peso] -= 2*epsilum
                        custo_menor = self.calcula_custos(dataset, results)
                        self.pesos_matriz[index_camada][index_neuronio][index_peso] += epsilum
                        gradiente_matriz[index_camada][index_neuronio].append((custo_maior - custo_menor)/(2*epsilum))
        print(gradiente_matriz)

    def atualiza_pesos(self, alpha):

        for index_camada in range(len(self.pesos_matriz)):
            for index_neuronio in range(len(self.pesos_matriz[index_camada])):
                self.bias_matriz[index_camada][index_neuronio] -= alpha*self.gradientes_bias[index_camada][index_neuronio]
                for index_peso in range(len(self.pesos_matriz[index_camada][index_neuronio])):
                    self.pesos_matriz[index_camada][index_neuronio][index_peso] -= alpha*self.gradientes[index_camada][index_neuronio][index_peso]
