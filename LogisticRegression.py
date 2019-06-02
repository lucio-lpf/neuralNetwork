import math
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

class LogisticRegression:
    def __init__(self, file):
        self.data = self.openDataframe(file)
        self.thetas = []

    def openDataframe(self, file):
        with open(file) as f:
            dataFrame = csv.reader(f)
            data = [r for r in dataFrame]
        minhaData = pd.DataFrame(data)
        a = minhaData.drop(columns=[1,2])
        b = minhaData.drop(columns=[0,2])
        c = minhaData.drop(columns=[0,1])
        minmaxScaler = preprocessing.MinMaxScaler()
        normalizeA = minmaxScaler.fit_transform(a)
        normalizeB = minmaxScaler.fit_transform(b)
        scal = MinMaxScaler(copy=True, feature_range=(-1, 1))
        print(scal.fit(normalizeA))
        print(scal.fit(normalizeB))
        normalizeA = scal.transform(normalizeA)
        normalizeB = scal.transform(normalizeB)
        dfa = pd.DataFrame(normalizeA)
        dfb = pd.DataFrame(normalizeB)
        s = pd.concat([dfa, dfb, c], axis=1)
        return s.values.tolist()

    #função sigmóide:
    def sigmoid(self, z):
        sig = 1/(1 + math.exp(-z))
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

    def functionInputs(self, theta0, theta1, theta2, data):
        x1 = float(data[0])
        x2 = float(data[1])
        fInput = theta0 + (x1*x1)*theta1 + (x2*x2)*theta2
        return self.sigmoid(fInput)

    #for neurons
    def functionInputsNeuron(self, thetas, data):
        fInput = 0
        for index, theta in thetas:
            theta*float(data[index])
        result = self.sigmoid(fInput)
        return result

    def diffPredClass(self, fInput, y):
        return fInput - y

    def gradient(self, thetaCurrent, alpha, dataFrame, predictions, correctClass, xi):
        alphaLen = alpha/len(dataFrame)
        sum = 0
        for index, data in enumerate(dataFrame):
            if xi == 9:
                sum = sum + self.diffPredClass(predictions[index], correctClass[index])
            else:
                sum = sum + (self.diffPredClass(predictions[index], correctClass[index])) * (float(data[xi])*float(data[xi]))
        return thetaCurrent - alphaLen * sum

    def trainning(self, alpha, dataframe, positiveClass, classificationIndex):
        theta0 = 0
        theta1 = 0
        theta2 = 0
        self.thetas.append(theta0)
        self.thetas.append(theta1)
        self.thetas.append(theta2)
        lastCost = 0
        currentCost = 0
        i = 0
        pred = []
        while lastCost - currentCost >= 0 or i < 3:
            lastCost = currentCost
            i = i + 1
            predictions = []
            pred = []
            correctClass = []
            for index, data in enumerate(dataframe):
                prediction = self.functionInputs(theta0, theta1, theta2, data)
                predictions.append(prediction)
                if prediction > 0.5:
                    pred.append(1.0)
                else:
                    pred.append(0.0)

                if data[classificationIndex] == positiveClass:
                    correctClass.append(1.0)
                else:
                    correctClass.append(0.0)
            cost = self.funcJ(correctClass, predictions)
            currentCost = cost
            newTheta0 = self.gradient(theta0, alpha, dataframe, predictions, correctClass, 9)
            newTheta1 = self.gradient(theta1, alpha, dataframe, predictions, correctClass, 0)
            newTheta2 = self.gradient(theta2, alpha, dataframe, predictions, correctClass, 1)
            theta0 = newTheta0
            theta1 = newTheta1
            theta2 = newTheta2
            print("custo: ", currentCost)
        return pred