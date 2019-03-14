#!/usr/bin/python3

import numpy as np


class Perceptron:

    def __init__(self, inputSize, learningRate, epochs):

        self.inputSize = inputSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights = np.random.rand(inputSize)

    def predict(self, dataX):
        dot = np.dot(self.weights, dataX.T)
        if dot > 0:
            dot = 1
        else:
            dot = 0
        return dot

    def train(self, dataX, dataY):

        for epoch in range(self.epochs):
            loss = 0
            accuracy = 0
            for inputX, labelY in zip(dataX, dataY):
                prediction = self.predict(inputX)
                self.weights[inputX] = self.learningRate * (labelY - prediction) * inputX.T
                loss += 0.5 * (labelY - prediction) ** 2
                if labelY - prediction == 0:
                    accuracy += 1

            print("Epoch:", epoch + 1, "Loss:", 0.5 * loss, "Accuracy:", accuracy / inputX.itemsize)


per = Perceptron(3, 0.00001, 5000)
dataX = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]])
dataY = np.array([[1], [0], [0], [0]])
print(per)
per.train(dataX, dataY)
print(per.predict(np.array([[1, 1, 1]])))
