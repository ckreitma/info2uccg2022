
import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Entrada
x = [0.7, 0.3, 0.1]

# Pesos W1
w1_1 = [0.2, 0.8, 0.2, 0.5]
w1_2 = [0.5, 0.2, 0.4, 0.99]
w1_3 = [0.3, 0.1, 0.2, 0.4]

# Importancia de las neuronas celestes
b1 = [0.8, 0.2, 0.5, 0.1]
h1 = w1_1[0]*x[0] + w1_2[0]*x[1] + w1_3[0]*x[2] + b1[0]
h2 = w1_1[1]*x[0] + w1_2[1]*x[1] + w1_3[1]*x[2] + b1[1]
h3 = w1_1[2]*x[0] + w1_2[2]*x[1] + w1_3[2]*x[2] + b1[2]
h4 = w1_1[3]*x[0] + w1_2[3]*x[1] + w1_3[3]*x[2] + b1[3]

print(f'Antes \t h1={h1:.4f}\th2={h2:.4f}\th3={h3:.4f}\th4={h4:.4f}\t')

h1 = sigmoid(h1)
h2 = sigmoid(h2)
h3 = sigmoid(h3)
h4 = sigmoid(h4)

print(f'Desp \t h1={h1:.4f}\th2={h2:.4f}\th3={h3:.4f}\th4={h4:.4f}\t')


# Pesos W2
w2_1 = [0.1, ]
w2_2 = [0.7, ]
w2_3 = [0.2, ]
w2_4 = [0.8, ]

b2 = [0.6]
k1 = h1*w2_1[0] + h2*w2_2[0] + h3*w2_3[0] + h4*w2_4[0] + b2[0]
print(f'Antes: {k1:.4f}')
k1 = sigmoid(k1)
print(f'Despu: {k1:.4f}')


m1 = np.array([
    [0.2, 0.8, 0.2, 0.5],
    [0.5, 0.2, 0.4, 0.99],
    [0.3, 0.1, 0.2, 0.4]
]
)

x1 = np.array([0.7, 0.3, 0.1])
print(f'{m1} {x1}')

b11 = np.array([0.8, 0.2, 0.5, 0.1])
r = np.dot(m1.T, x1) + b11

print(f'Resultado = {r}')
