import math
import random
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


class Neuron:
    def __init__(self, id, inputs, positiveClass, classificationIndex, numeroDeThetas):
        self.activation = 0
        self.thetas = []
        self.inputs = inputs
        for index in range(0, numeroDeThetas):
                self.thetas.append(random.uniform(0, 1))
        print("numero de thetas: ", len(self.thetas))
        self.dataframe = None
        self.predictions = None
        self.correctClass = []
        self.alpha = None
        self.positiveClass = positiveClass
        self.classificationIndex = classificationIndex
        self.delta = None
        self.classif = None
        self.gradient = []
        self.deltaSeguinte = [0]*len(self.thetas)
        if inputs[classificationIndex] == positiveClass:
            self.classif = 1.0
        else:
            self.classif = 0
        self.cost = 0
        print("numero de inputs", len(inputs))
        print("inicializando neuronio")

    # for neurons
    def neuronActivation(self):
        fInputs = []
        for index, theta in enumerate(self.thetas):
            fInputs.append(self.thetas[index]*self.inputs[index])
        value = sum(fInputs)
        self.activation = self.sigmoid(value)
        return self.sigmoid(value)

    # função sigmóide:
    def sigmoid(self, z):
        sig = 1 / (1 + math.exp(-z))
        return sig

    # if 1 = true; if 0 = false
    def calculateError(self, y, z):
        if y == 0:
            if 1-z == 0:
                value = 999999
            else:
                value = -math.log10(1.0 - z)
        else:
            if z == 0:
                value = 999999
            else:
                value = -math.log10(z)
        return value

    # para achar os menores valores de tetas
    # ys = classe correta
    # zs = classe predita
    def funcJ(self, ys, zs):
        cost = 0
        for index in range(0,len(zs)):
            error = self.calculateError(ys[index], zs[index])
            cost = cost + error
        return cost/len(zs)

    def functionInputs(self, datas):
        xis = []
        for index, theta in enumerate(self.thetas):
            xis.append(datas[index]*theta)
        fInput = sum(xis)
        fInput = self.activation
        return self.sigmoid(fInput)

    def diffPredClass(self, fInput, y):
        return fInput - y

    def gradientz(self, thetaCurrent, alpha, dataFrame, predictions, correctClass, xi):
        alphaLen = alpha/len(dataFrame)
        sum = 0
        for index, data in enumerate(dataFrame):
            sum = sum + (self.diffPredClass(predictions[index], correctClass[index])) * (float(data[xi]))
        return thetaCurrent - alphaLen * sum

    def training(self, dataframe):
        self.predictions = []
        self.correctClass = []
        for index, data in enumerate(dataframe):
            prediction = self.functionInputs(data)
            self.predictions.append(prediction)
            if data[self.classificationIndex] == self.positiveClass:
                self.correctClass.append(1.0)
            else:
                self.correctClass.append(0.0)
        self.cost = self.funcJ(self.correctClass, self.predictions)
        return self.cost

    def calcDelta(self, deltaActivations):
        if len(deltaActivations) != 0:
            self.delta = sum(deltaActivations) * self.activation * (1 - self.activation)
        else:
            self.delta = self.activation - self.classif
        return self.delta

    def calcGradient(self, deltaCamadaBefore, dataframe):
        self.gradient = []
        for index, delta in enumerate(deltaCamadaBefore):
            self.gradient.append(delta * self.activation)
        self.adjustWeights()
        cost = self.training(dataframe)
        return self.gradient

    def adjustWeights(self):
        for index, theta in enumerate(self.thetas):
            self.thetas[index] = self.thetas[index] - self.activation * self.gradient[index] * self.cost