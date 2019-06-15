from LogisticRegression import LogisticRegression
from NeuralNetwork import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as mt
import csv
from Neuron import Neuron

alpha = 0.01

#treina o modelo

wine = LogisticRegression("./datasets/wine.csv")
wine.normalizeData()

# camadas = [3, 2, 1]
# entradas = len(wine.data[0][0])
# nn = NeuralNetwork(entradas, camadas)
# nn.calcula_saidas(wine.data[0])

