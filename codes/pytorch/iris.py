import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# IN BEST 98,10% Training Acc
# IN BEST 100,00% Test Acc

# Definicja przykładowej funkcji aktywacji

class Heavyside(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, input):
		for i in range(len(input)):
			if input[i] > 0:
				input[i] = 1
			else:
				input[i] = 0
		return input


# Inicjalizacja funkcji aktywacji

heavy = Heavyside()

# Definicja naszej sieci (perceptronu)


class Network(nn.Module):
	def __init__(self, input_dim):
		super(Network, self).__init__()
		self.first_layer = nn.Linear(input_dim, 6)
		self.out_layer = nn.Linear(6, 1)

	def forward(self, x):
		out = self.first_layer(x)
		out = F.relu(out)
		out = self.out_layer(out)
		out = F.relu(out)
		return out


# Sprawdzenie czy czy jest dostępna karta graficzna z cudą
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# Wczytanie datasetu

iris = load_iris()

# Stworzenie tabeli danych

data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Utworzenie datasetu

x = np.array(data.drop('target', axis=1))
y = np.array(data['target'])

# Podział na dane uczące i testowe

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7)

# Zamiana numpy array na tensor

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)

# Podobnie jak w przypadku x

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Inicjalizacja sieci

net = Network(4)

# Ustawienie modelu w tryb uczenia
net.train()

# Wysłanie modelu do device
net.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)


# Wypiszmy parametry sieci
for parameter in net.parameters():
	print(parameter)

for i in range(40):
	train_accuracy = 0
	print('Epoch:[{}\{}]'.format(i+1, 40))
	for input, target in zip(x_train, y_train):

		# Zerujemy gradienty
		optimizer.zero_grad()

		input = input.float().to(device)
		target = target.float().to(device)

		# Predykcja
		output = net(input)

		# Obliczenie kosztu
		loss = criterion(output, target)

		# Krok wstecznej propagacji
		loss.backward()

		# Aktualizacja wag
		optimizer.step()

		if (target - output) < 0.5:
			train_accuracy += 1

		# print('Predicted:{}, Expected:{}'.format(output.item(), target))
	print('Loss:{:.2f}, Accuracy:{:.2f}%'.format(loss.item(), train_accuracy / len(y_train) * 100))
	

# Przełączenie sieci w tryb ewaluacji
net.eval()

net.to(device)

# Testowanie sieci na danych uczących UWAGA TO JEST ZŁY NAWYK

validation_accuracy = 0
for data, target in zip(x_test, y_test):

	data = data.float().to(device)
	target = target.float().to(device)

	predicted = net(data).item()
	if target - predicted < 0.5:
		validation_accuracy += 1

	# print('Predicted:{}, Expected:{}'.format(predicted, target.item()))
print('Train accuracy:{:.2f}%, Validation accuracy:{:.2f}%'.format(train_accuracy / len(y_train) * 100, validation_accuracy / len(y_test) * 100))
