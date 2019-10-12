import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Definicja naszej funkcji aktywacji

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
		self.first_layer = nn.Linear(input_dim, 1)

	def forward(self, x):
		out = self.first_layer(x)
		out = heavy(out)
		return out


# Sprawdzenie czy czy jest dostępna karta graficzna z cudą
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# Utworzenie datasetu

x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

# Zamiana numpy array na tensor
x = torch.from_numpy(x)

# Podobnie jak w przypadku x
y = np.array([[1], [0], [0], [0]])

y = torch.from_numpy(y)

# Inicjalizacja sieci

net = Network(2)

# Ustawienie modelu w tryb uczenia
net.train()

# Wysłanie modelu do device
net.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)


# Wypiszmy parametry sieci
for parameter in net.parameters():
	print(parameter)

for i in range(100):
	accuracy = 0
	print('Epoch:[{}\{}]'.format(i+1, 100))
	for input, target in zip(x, y):

		# Zerujemy gradienty
		optimizer.zero_grad()

		input = input.float()
		target = target.float()

		input = input.to(device)
		target = target.to(device)

		# Predykcja
		output = net(input)

		# Obliczenie kosztu
		loss = criterion(output, target)

		# Krok wstecznej propagacji
		loss.backward()

		# Aktualizacja wag
		optimizer.step()

		if output == target:
			accuracy += 1

	print('Loss:', loss.item(), 'Accuracy:', str(accuracy / len(y) * 100) + '%')

# Przełączenie sieci w tryb ewaluacji
net.eval()

net.to(device)

# Testowanie sieci na danych uczących UWAGA TO JEST ZŁY NAWYK
for data, target in zip(x, y):

	data = data.float()
	data = data.to(device)

	target = target.float()
	taget = target.to(device)

	print('Predicted:{}, Expected:{}'.format(net(data).item(), target.item()))
