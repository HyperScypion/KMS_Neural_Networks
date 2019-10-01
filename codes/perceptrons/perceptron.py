#!/usr/bin/python3

import numpy as np


class Perceptron:

    def __init__(self, inputSize, learningRate, epochs):

        self.inputSize = inputSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights = np.zeros(inputSize + 1)

    def activ_func(self, dot):
        if dot > 0:
            dot = 1
        else:
            dot = 0
        return dot

    def forward(self, dataX):
        dot = np.dot(dataX, self.weights[1:]) + self.weights[0]
        dot = self.activ_func(dot)
        return dot

    def train(self, x_train, y_train):

        for epoch in range(self.epochs):
            loss = 0
            accuracy = 0
            for data, label in zip(x_train, y_train):
                prediction = self.forward(data)
                self.weights[1:] += self.learningRate * (label - prediction) * data
                self.weights[0] += self.learningRate * (label - prediction)
                loss += 0.5 * (label - prediction) ** 2
                if int(label) - prediction == 0:
                    accuracy += 1

                print('Epoch {}, [{}/{}]'.format(epoch, prediction, int(label)))
            print(accuracy / len(y_train) * 100.)
    
    def predict(self, x):
        return self.forward(x)

per = Perceptron(2, 0.01, 100)
dataX = np.array([[1, 1], [1, 0], [0, 0], [0, 1]])
dataY = np.array([[1], [0], [0], [0]])
per.train(dataX, dataY)
print(per.predict(np.array([[1, 1]])))
print(per.predict(np.array([[1, 0]])))
print(per.predict(np.array([[0, 1]])))
print(per.predict(np.array([[0, 0]])))
