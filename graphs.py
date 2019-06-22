import numpy as np
import matplotlib.pyplot as plt

class Graphs:
    def __init__(self):
        print("inicializa gráficos")

    def f1score(self, confusion_matrix, enfase):
        vps = []
        fns = []
        fps = []

        for i in range(len(confusion_matrix)):
            vps.append(0)
            fns.append(0)
            fps.append(0)

        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                if i == j:
                    vps[i] = confusion_matrix[i][j]
                else:
                    fns[i] += confusion_matrix[i][j]
                    fps[i] += confusion_matrix[j][i]


        recalls = []
        precs = []
        for i in range(len(vps)):
            if vps[i] + fns[i] == 0:
                recalls.append(0)
            else:
                recalls.append(vps[i]/(vps[i] + fns[i]))

            if vps[i] + fps[i] == 0:
                precs.append(0)
            else:
                precs.append(vps[i]/(vps[i] + fps[i]))

        f1scores = []
        for i in range(len(vps)):
            pxr = precs[i] * recalls[i]
            bxpxr = (enfase * enfase * precs[i]) + recalls[i]
            if bxpxr != 0:
                f1scores.append((1 + enfase * enfase) * (pxr/bxpxr))
            else:
                f1scores.append(0)

        print("F1-SCORES:", f1scores)




    def classificacao(self, resultados, saida_da_rede, epocas, enfase_f1_score):
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
        x = ("Numero de epocas: %d" % epocas)
        ax.set_title(x)
        fig.tight_layout()
        plt.show()
        self.f1score(confusion_matrix, enfase_f1_score)