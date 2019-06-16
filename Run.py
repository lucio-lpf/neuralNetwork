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

wine = LogisticRegression("./datasets/teste.csv")
wine.normalizeData()

camadas = [3, 2, 1]
entradas = len(wine.data[0]) - 1
nn = NeuralNetwork(entradas, camadas)


nn.treina_rede(wine.data[0], wine.results[0])

#for index in range(len(wine.data)):
#    nn.treina_rede(wine.data[index], wine.results[index])

