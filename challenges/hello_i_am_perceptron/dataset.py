import os
import random
import numpy as np
import matplotlib.pylab as plt


class Dataset:
	def __init__(self, path='.'):
		self.path = path


	def preprocess_img(self, img):
		label = img[2]
		array = []
		img = plt.imread(self.path + img)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if img[i, j].any() == 1:
					array.append(1)
				else:
					array.append(0)

		array = np.array(array)
		return array, label		

	def create_dataset(self):
		self.dataset = []
		dir = os.listdir(self.path)
		for i in dir:
			img, label = self.preprocess_img(i)
			self.dataset.append((img, label))

		self.dataset = np.array(self.dataset)
		return self.dataset
	
	def random_sample(self):
		return random.choice(self.dataset)
	
