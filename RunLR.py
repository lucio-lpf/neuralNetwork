from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd

#treina o modelo
lr = LogisticRegression("teste.csv")
pred = lr.trainning(0.1, lr.data, "sim", 2)


myData = pd.DataFrame(lr.data)
x1 = myData[0].tolist()
x2 = myData[1].tolist()
#print(pred)
def pltcolor(lst):
    cols=[]
    for index in range(0, len(lst)):
        if myData[2][index] == "nao" and pred[index] == 0.0:
            cols.append('blue')
        elif myData[2][index] == "sim" and pred[index] == 1.0:
            cols.append('red')
        elif myData[2][index] == "sim" and pred[index] == 0.0:
            cols.append('pink')
        elif myData[2][index] == "nao" and pred[index] == 1.0:
            cols.append('purple')

    return cols
cols = pltcolor(x1)

plt.scatter(x=x1, y=x2, s=80, c=cols)

#x = np.linspace(-1,1) # 100 linearly spaced numbers
#y = -(lr.thetas[0] + lr.thetas[1]*(x))/lr.thetas[2]

#plt.plot(y,x) # 2*sin(x)/x and 3*sin(x)/x
plt.show() # show the plot