from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from Neuron import Neuron

#treina o modelo
lr = LogisticRegression("teste.csv")
pred = lr.trainning(0.001, lr.data, "sim", 2)

#neuron = Neuron(0, [0.4, 0.8])
#data = [-0.4, 0.8]
#r = neuron.neuronActivation(data)
#print(r)




myData = pd.DataFrame(lr.data)
x1 = myData[0].tolist()
x2 = myData[1].tolist()
print(pred)
def pltcolor(lst):
    cols=[]
    for index in range(0, len(lst)):
        if myData[2][index] == "nao" and pred[index] == 0.0:
            cols.append('blue')
        elif myData[2][index] == "sim" and pred[index] == 1.0:
            cols.append('red')
        else:
            cols.append('green')

    return cols
cols = pltcolor(x1)


plt.scatter(x=x1, y=x2, s=100, c=cols)
plt.show()