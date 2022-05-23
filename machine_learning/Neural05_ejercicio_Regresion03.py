# -*- coding: utf-8 -*-
from scipy import signal
import numpy as np
import math
import matplotlib.pyplot as plt

cantidad_puntos = 400
cantidad_iteraciones = 50000
mostrar_cada = 5000
grado = 9

# Create random input and output data
# Una lista de puntos entre -pi y +pi
x = np.linspace(-1, 1, cantidad_puntos)
lista_y = []
# for v in x:
lista_y = signal.sawtooth(0.01 * np.pi * 5 * x)
# print(f'valor={valor}')
# lista_y.append(valor)
y = np.array(lista_y)

# plt.plot(x, y)
# plt.ylim(-10, 10)
# plt.show()
# quit()


# Randomly initialize weights
pesos = []
for g in range(grado):
    pesos.append(np.random.randn())


print(f'Inicial = {pesos}')


# UCCG. Agregamos la lista de errores para ir visualizando.
error_history = []
epoch_list = []


def prediction(pesos, x):
    factor = 0
    resultado = 0
    for i in range(len(pesos)):
        resultado += pesos[i]*x**factor
        factor += 1
    return resultado


# Razón de aprendizaje
learning_rate = 1e-6

for t in range(cantidad_iteraciones):

    # Forward pass: compute predicted y
    y_pred = prediction(pesos, x)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()

    # Agregamos el control del historial de error.
    error_history.append(loss)
    epoch_list.append(t)

    # Cada veinte iteraciones vamos mostrando los avances.
    if t % mostrar_cada == 0 or t == (cantidad_iteraciones-1):
        print(t, loss)

        # Ahora vamos a la prediccion
        plt.figure(figsize=(40, 15))
        plt.plot(x, y)
        titulo = f'Iteracion={t} {pesos}'
        plt.title(titulo)
        plt.xlabel('X')
        plt.ylabel('Y')

        diente_aproximado = []
        for p in x:
            diente_aproximado.append(prediction(pesos, p))
        plt.plot(x, diente_aproximado)

        # Fijamos el límite y.
        plt.ylim(-2, 2)
        plt.show()

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    gradiente = []
    for factor in range(len(pesos)):
        gradiente.append((grad_y_pred*x**factor).sum())
        factor += 1

    # Update weights
    for i in range(len(pesos)):
        pesos[i] -= (learning_rate * gradiente[i])

# print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

# Imprimimos el avance.
# plt.figure(figsize=(30,15))
# plt.plot(epoch_list,error_history)
# plt.xlabel('Epoch')
# plt.ylabel('Erro')
# plt.show()

# plt.figure(figsize=(40, 15))
# plt.plot(x, y)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
