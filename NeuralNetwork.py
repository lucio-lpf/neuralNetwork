import numpy.matlib as np


class NeuralNetwork:

    def __init__(self, entradas, camadas):

        print("inicializando matriz de pesos da rede neural")
        self.weights_matrix = [[None for x in range(camadas[y])] for y in range(len(camadas))]
        for index, camada in enumerate(camadas):
            for index_j in range(0, camadas[index]):
                if index is 0:
                    self.weights_matrix[index][index_j] = [1 for _ in range(0, entradas)]
                else:
                    self.weights_matrix[index][index_j] = [1 for _ in range(0, camadas[index - 1])]

        print("Pesos das camadas:")
        for index, line in enumerate(self.weights_matrix):
            print(line)

    def calcula_saidas(self, registro):

        matriz_de_saidas = [[0 for x in range(len(self.weights_matrix[y]))] for y in range(len(self.weights_matrix))]

        result = registro.pop()

        for index in range(0, len(matriz_de_saidas)):
            if index is 0:
                for index_j in range(0, len(matriz_de_saidas[index])):
                    matriz_de_saidas[0][index_j] = np.matmul(registro, self.weights_matrix[index][index_j])
            else:
                for index_j in range(0, len(matriz_de_saidas[index])):
                    matriz_de_saidas[index][index_j] = np.matmul(matriz_de_saidas[index - 1], self.weights_matrix[index][index_j])
        for i in range(0, len(matriz_de_saidas)):
            for j in range(0, len(matriz_de_saidas[i])):
                print("Saída do nueronio: ", j, " da camada ", i, "é: ", matriz_de_saidas[i][j])

    def corrige_pesos(self, matriz_de_saidas, resultado):
        pass