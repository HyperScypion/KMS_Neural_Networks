#!/usr/bin/python3

import random
import numpy as np


class Perceptron:
	def __init__(self, l_alg='spla', epochs=1000, l_rate=0.1):
		self.l_alg = l_alg
		self.epochs = epochs
		self.l_rate = l_rate

	def forward(self, input):
		if np.dot(self.__weights[1:], input) + self.__weights[0] >= 0:
			return 1
		else:
			return 0

	def fit(self, x_train, y_train):
		self.__weights = np.random.randn(x_train.shape[1] + 1)
		if self.l_alg == 'spla':
			self.spla(x_train, y_train)
		elif self.l_alg == 'pla':
			self.pla(x_train, y_train)
		elif self.l_alg == 'pla_r':
			self.pla_r(x_train, y_train)
		else:
			print('Wrong learning algorithm! Exiting...')
			exit(-1)


	def _weights(self):
		return self.__weights

	def _update_weights(self, prediction, label, data):
		self.__weights[1:] += self.l_rate * (label - prediction) * data
		self.__weights[0] += self.l_rate * (label - prediction)

	def spla(self, x_train, y_train):
		error = 1
		while error != 0:

			index = random.randrange(len(x_train))
			prediction = self.forward(x_train[index])
			if y_train[index] - prediction == 0:
				continue
			else:
				self._update_weights(prediction, y_train[index], x_train[index])
			
			error = 0
			for data, label in zip(x_train, y_train):
				if label - self.forward(data) != 0:
					error += 1

	def pla(self, x_train, y_train):
		l_time = 0
		current_best = 0
		pocket = self.__weights
		for epoch in range(self.epochs):
			index = random.randrange(len(x_train))
			prediction = self.forward(x_train[index])
			if y_train[index] - prediction == 0:
				l_time += 1
				if l_time > current_best:
					current_best = l_time
					pocket = self.__weights
					continue
			else:
				self._update_weights(prediction, y_train[index], x_train[index])
				l_time = 0

	def pla_r(self, x_train, y_train):
		l_time = 0
		acc = 0
		current_best = 0
		best_acc = 0
		pocket = self.__weights
		for epoch in range(self.epochs):
			index = random.randrange(len(x_train))
			prediction = self.forward(x_train[index])
			if y_train[index] - prediction == 0:
				l_time += 1
				for data, label in zip(x_train, y_train):
					if label - self.forward(data) == 0:
						acc += 1

				if acc > best_acc and l_time > current_best:
					best_acc = acc
					current_best = l_time
					pocket = self.__weights

			else:
				self._update_weights(prediction, y_train[index], x_train[index])
				acc = 0
				l_time = 0

			
# per = Perceptron('pla', 90, 0.1)
# per = Perceptron()
per = Perceptron('pla_r')

data_X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
data_Y = np.array([[1], [0], [0], [0]])

per.fit(data_X, data_Y)
print('fitted')
print(per.forward(np.array([1, 1])))
print(per.forward(np.array([1, 0])))
print(per.forward(np.array([0, 1])))
print(per.forward(np.array([0, 0])))
print(per._weights())

