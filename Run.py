from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as mt
from Neuron import Neuron

#treina o modelo
lr = LogisticRegression("teste.csv")


n21 = Neuron(1, lr.data[0], "sim", 2, 2)
n22 = Neuron(1, lr.data[0], "sim", 2, 2)
n23 = Neuron(1, lr.data[0], "sim", 2, 2)
n24 = Neuron(1, lr.data[0], "sim", 2, 2)

n31 = Neuron(2, lr.data[0], "sim", 2, 4)
n32 = Neuron(2, lr.data[0], "sim", 2, 4)

n4 = Neuron(3, lr.data[0], "sim", 2, 2)
cost = 0.5
while (cost > 0.1):
    for data in lr.data:
        n21.inputs = data
        n22.inputs = data
        n23.inputs = data
        n24.inputs = data


        a21 = n21.neuronActivation()
        a22 = n22.neuronActivation()
        a23 = n23.neuronActivation()
        a24 = n24.neuronActivation()

        n31.inputs = [a21, a22, a23, a24, data[2]]
        n32.inputs = [a21, a22, a23, a24, data[2]]

        a31 = n31.neuronActivation()
        a32 = n32.neuronActivation()

        n4.inputs = [a31, a32, data[2]]

        #saida da rede
        a4 = n4.neuronActivation()

        #backpropagation
        d4 = n4.calcDelta([])
        print("erro : ", d4)


        cost = n4.training(lr.data)
        #print("custo : ", cost)

        d31 = n31.calcDelta([d4*n4.thetas[0]])
        d32 = n31.calcDelta([d4*n4.thetas[1]])

        d21 = n21.calcDelta([d31*n31.thetas[0], d32*n32.thetas[0]])
        d22 = n22.calcDelta([d31*n31.thetas[1], d32*n32.thetas[1]])
        d23 = n23.calcDelta([d31*n31.thetas[2], d32*n32.thetas[2]])
        d24 = n24.calcDelta([d31*n31.thetas[3], d32*n32.thetas[3]])


        grad31 = n31.activation*d4
        grad32 = n32.activation*d4

        #change tethas
        n4.thetas[0] = n4.thetas[0] - 0.1 * grad31 * cost
        n4.thetas[1] = n4.thetas[1] - 0.1 * grad32 * cost

        grad211 = n21.activation * d31
        grad221 = n22.activation * d31
        grad231 = n23.activation * d31
        grad241 = n24.activation * d31

        n31.thetas[0] = n31.thetas[0] - 0.1 * grad211 * cost
        n31.thetas[1] = n31.thetas[1] - 0.1 * grad221 * cost
        n31.thetas[2] = n31.thetas[2] - 0.1 * grad231 * cost
        n31.thetas[3] = n31.thetas[3] - 0.1 * grad241 * cost


        grad212 = n21.activation * d32
        grad222 = n22.activation * d32
        grad232 = n23.activation * d32
        grad242 = n24.activation * d32

        n32.thetas[0] = n32.thetas[0] - 0.1 * grad212 * cost
        n32.thetas[1] = n32.thetas[1] - 0.1 * grad222 * cost
        n32.thetas[2] = n32.thetas[2] - 0.1 * grad232 * cost
        n32.thetas[3] = n32.thetas[3] - 0.1 * grad242 * cost


