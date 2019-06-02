import math

class Neuron:
    def __init__(self, id, inputs):
        self.id = id
        self.ativation = 0
        self.thetas = []
        for input in inputs:
            self.thetas.append(0)
        print("inicializando neuronio")

    # for neurons
    def neuronActivation(self, data):
        fInput = 0
        for index, theta in enumerate(self.thetas):
            theta * float(data[index])
        result = self.sigmoid(fInput)
        return result

    # função sigmóide:
    def sigmoid(self, z):
        sig = 1 / (1 + math.exp(-z))
        return sig







