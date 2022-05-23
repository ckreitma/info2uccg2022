# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

cantidad_puntos = 400
cantidad_iteraciones = 10000
mostrar_cada = 5000

# Create random input and output data
# Una lista de puntos entre -pi y +pi
x = np.linspace(-math.pi, math.pi, cantidad_puntos)

# Valores verdaderos del "seno"
y = np.sin(x)

#plt.plot(x, y)
#plt.ylim(-10, 10)
# plt.show()
# quit()

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

print(f'Inicial: y = {a} + {b} x + {c} x^2 + {d} x^3')


# UCCG. Agregamos la lista de errores para ir visualizando.
error_history = []
epoch_list = []


def prediction(a, b, c, d, x):
    # y = a + b x + c x^2 + d x^3
    return a + b * x + c * x ** 2 + d * x ** 3


# Razón de aprendizaje
learning_rate = 1e-6

for t in range(cantidad_iteraciones):

    # Forward pass: compute predicted y
    y_pred = prediction(a, b, c, d, x)

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
        titulo = 'Iteracion=' + str(t) + ' y = ' + str(a) + ' + ' + str(b) + 'x ' + str(c) + 'x^2  ' + str(d) + 'x^3'
        plt.title(titulo)
        plt.xlabel('X')
        plt.ylabel('Y')

        seno_aproximado = []
        for p in x:
            seno_aproximado.append(prediction(a, b, c, d, p))
        plt.plot(x, seno_aproximado)

        # Fijamos el límite y.
        plt.ylim(-2, 2)
        plt.show()

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

# Imprimimos el avance.
# plt.figure(figsize=(30,15))
# plt.plot(epoch_list,error_history)
# plt.xlabel('Epoch')
# plt.ylabel('Erro')
# plt.show()

#plt.figure(figsize=(40, 15))
#plt.plot(x, y)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
