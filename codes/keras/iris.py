import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



# Wczytanie datasetu

iris = load_iris()

# Stworzenie tabeli danych

data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Wyodrębnienie danych

x = data.drop('target', axis=1)
y = data['target']

# Podział na dane uczące i testowe

# Definicja naszego modelu

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7)

model = Sequential()
model.add(Dense(6, input_dim=4, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

with tf.device('/device:GPU:0'):

    # Trenowanie modelu

    model.fit(x_train, y_train, epochs=40)


loss, accuracy = model.evaluate(x_test, y_test)
print('Loss: {}, accuracy: {}'.format(loss, accuracy))
