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
        ativacao_matriz = deepcopy(self.calcula_saidas(atributos))
        saida_da_rede = ativacao_matriz[len(ativacao_matriz) - 1]

        delta_matriz = self.calcula_deltas(ativacao_matriz, resultado)
        gradientes_matriz = self.calcula_gradientes(atributos, ativacao_matriz, delta_matriz)
        gradientes_matriz_bias = deepcopy(delta_matriz)

        print("matriz gradiente", gradientes_matriz)
        print("gradiente bias:", gradientes_matriz_bias)
        print("==============================")
        saidas_da_rede = []
        for data in dataset:
            saidas = deepcopy(self.calcula_saidas(atributos))
            saida = saidas[len(saidas) - 1]
            saidas_da_rede.append(saida)
        taxa_regularizacao_custo = self.calcula_taxa_regularizacao(len(dataset))
        print(taxa_regularizacao_custo)
        custo = sum(self.funcao_custo_J(dataset, results, saidas_da_rede)) + taxa_regularizacao_custo
        print(custo)

        len_matriz = len(pesos_mat)
        for index in range(len_matriz):
            gradientes = gradientes_matriz[index]
            pesos = pesos_mat[index]
            pesos_mat[index] = self.atualizacao_do_peso(pesos, gradientes, alfa, custo)
        for i in range(len(self.bias_matriz)):
            for j in range(len(self.bias_matriz[i])):
                self.bias_matriz[i][j] -= alfa*delta_matriz[i][j]*custo

        self.pesos_matriz = pesos_mat




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
            print(saidas_matriz[index_camada_saidas][index_delta])
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

    def delta_camada_saida(self, saidas_rede, saidas_esperada):
        delta = [None] * (len(saidas_rede))
        for index in range(len(saidas_rede)):
            delta[index] = saidas_rede[index] - float(saidas_esperada[index])
        return delta

    def delta_camadas_ocultas(self, thetas_ligacao, deltas_anteriores, ativacao_do_neuronio):
        deltas = [0] * (len(ativacao_do_neuronio))
        for index in range(0, len(deltas)):
            for index2 in range(0, len(thetas_ligacao)):
                theta = thetas_ligacao[index2][index]
                delta = deltas_anteriores[index2]
                deltas[index] = deltas[index] + theta*delta
            deltas[index] = deltas[index] * ativacao_do_neuronio[index] * (1 - ativacao_do_neuronio[index])
        return deltas

    def gradientes_do_peso(self, ativacao, delta_camada_anterior, pesos_da_camada):
        #gradiente = [[0] * (len(ativacao))] * (len(delta_camada_anterior))
        gradiente = []
        for i1 in range(len(delta_camada_anterior)):
            gradiente.append([])
            for i2 in range(len(ativacao)):
                gradiente[i1].append(0)

        for index in range(len(ativacao)):
            #gradiente.append([])
            for index2 in range(len(delta_camada_anterior)):
                atv = ativacao[index]
                delta = delta_camada_anterior[index2]
                gradiente[index2][index] = (atv * delta) + self.fator_regularizacao*pesos_da_camada[index2][index]
                #gradiente[index].append(atv * delta)
        return gradiente

    def funcao_custo_J(self, dataset, resultados_certos, saidas_funcao):
        n = len(dataset)
        custos_J = [0] * (len(resultados_certos[0]))
        for index in range(0, len(dataset)):
            for index2 in range(len(resultados_certos[index])):
                s = saidas_funcao[index][index2]
                log_saida = math.log10(s)
                log_um_menos_saida = math.log10(1 - s)
                resultado = resultados_certos[index][index2]
                custos_J[index2] = custos_J[index2] - resultado*log_saida - (1 - resultado)*log_um_menos_saida

        for index in range(len(custos_J)):
            custos_J[index] = (1/n) * custos_J[index]
        return custos_J

    def atualizacao_do_peso(self, pesos, gradientes, alfa, custo):
        pesos_atualizados = deepcopy(pesos)
        for index in range(len(pesos)):
            for index2 in range(len(pesos[index])):
                p = pesos[index][index2]
                g = gradientes[index][index2]
                pesos_atualizados[index][index2] = p - alfa * g * custo
        return pesos_atualizados
