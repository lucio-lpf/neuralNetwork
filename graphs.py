import numpy as np
import matplotlib.pyplot as plt

class Graphs:
    def __init__(self):
        print("inicializa gráficos")


    def classificacao(self, resultados, saida_da_rede):
        confusion_matrix = []
        resultados_certos = resultados
        saidas_funcao = saida_da_rede

        for i in range(len(resultados_certos[0])):
            confusion_matrix.append([])
            for j in range(len(resultados_certos[0])):
                confusion_matrix[i].append(0)

        for i in range(0, len(resultados_certos)):
            for j in range(0, len(resultados_certos[0])):
                if resultados_certos[i][j] > 0:
                    maximum = max(saidas_funcao[i])
                    for w in range(len(saidas_funcao[i])):
                        if maximum == saidas_funcao[i][w]:
                            confusion_matrix[j][w] += 1

        coluna = []
        linha = []
        for i in range(len(confusion_matrix[0])):
            argument = "classe: ", i
            coluna.append(argument)
            linha.append(argument)

        harvest = np.array(confusion_matrix)

        fig, ax = plt.subplots()
        im = ax.imshow(harvest)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(linha)))
        ax.set_yticks(np.arange(len(coluna)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(linha)
        ax.set_yticklabels(coluna)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(coluna)):
            for j in range(len(linha)):
                text = ax.text(j, i, harvest[i, j],
                               ha="center", va="center", color="w")

        ax.set_title("Harvest of local linha (in tons/year)")
        fig.tight_layout()
        plt.show()