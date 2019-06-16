import numpy.matlib as np
import math
from random import randint


class NeuralNetwork:

    def __init__(self, entradas, camadas):

        print("inicializando matriz de pesos da rede neural")
        self.pesos_matriz = [[None for x in range(camadas[y])] for y in range(len(camadas))]
        for index, camada in enumerate(camadas):
            for index_j in range(0, camadas[index]):
                if index is 0:
                    self.pesos_matriz[index][index_j] = [randint(1, 9) for _ in range(0, entradas)]
                else:
                    self.pesos_matriz[index][index_j] = [randint(1, 9) for _ in range(0, camadas[index - 1])]

        print("Pesos das camadas:")
        for index, line in enumerate(self.pesos_matriz):
            print(line)

    def treina_rede(self, atributos, resultado):
        pesos_matriz = self.pesos_matriz
        ativacao_matriz = self.calcula_saidas(atributos)
        saida_da_rede = ativacao_matriz[len(ativacao_matriz) - 1]
        delta_matriz = ativacao_matriz

        for index in reversed(range(len(delta_matriz))):
            if index == len(delta_matriz) - 1:
                delta_matriz[index] = self.delta_camada_saida(saida_da_rede, resultado)
            else:
                delta_matriz[index] = self.delta_camadas_ocultas(pesos_matriz[index + 1], delta_matriz[index + 1], ativacao_matriz[index])

        gradientes_matriz = pesos_matriz
        for index in range(len(gradientes_matriz)):
            for index2 in range(len(gradientes_matriz[index])):
                print("ativacao matriz: ", ativacao_matriz[index])
                print("delta matriz: ", delta_matriz[index])
                print("matriz gradientes:", gradientes_matriz[index])
        print("matriz inteira: ", gradientes_matriz)




    def calcula_saidas(self, registro):

        matriz_de_saidas = [[0 for x in range(len(self.pesos_matriz[y]))] for y in range(len(self.pesos_matriz))]

        result = registro.pop()

        for index in range(0, len(matriz_de_saidas)):
            if index is 0:
                for index_j in range(0, len(matriz_de_saidas[index])):
                    matriz_de_saidas[0][index_j] = self.sigmoide(np.matmul(registro, self.pesos_matriz[index][index_j]))
            else:
                for index_j in range(0, len(matriz_de_saidas[index])):
                    matriz_de_saidas[index][index_j] = self.sigmoide(np.matmul(matriz_de_saidas[index - 1], self.pesos_matriz[index][index_j]))
        for i in range(0, len(matriz_de_saidas)):
            for j in range(0, len(matriz_de_saidas[i])):
        return matriz_de_saidas
                print("Saída do nueronio: ", j, " da camada ", i, "é: ", matriz_de_saidas[i][j])

    def corrige_pesos(self, matriz_de_saidas, resultado):
        pass

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

    def gradientes_do_peso(self, ativacao, delta_camada_anterior):
        return ativacao * delta_camada_anterior

    def funcao_custo_J(self, dataset, resultados_certos, saidas_funcao):
        n = len(dataset)
        sum = 0
        for index in range(0, len(dataset)):
            log_saida = math.log10(saidas_funcao[index])
            log_um_menos_saida = math.log10(1 - saidas_funcao[index])
            resultado = resultados_certos[index]
            sum = sum - resultado*log_saida - (1 - resultado)*log_um_menos_saida
        custo_J = (1/n) * sum
        return custo_J

    def atualizacao_do_peso(self, peso_atual, alfa, gradiente, custo):
        return peso_atual - alfa*gradiente*custo

