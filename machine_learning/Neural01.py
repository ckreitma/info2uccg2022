# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
import numpy as np  # helps with the math
import matplotlib.pyplot as plt  # to plot error during training

# input data
# Este dato está siendo realmente una forma de entender los pesos, ya que
# por la inspeccion de los datos de entrenamiento, el único que realmente
# importa para la salida es el primer bit de entrada (se ignoran los demás)
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0],  [0], [0],  [1], [1], [1]])

inputs_ucg = np.array([
    [0, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
])

outputs_ucg = np.array([[1], [1], [0], [1], [1], [0], [0], [0], [1]])


# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.weights1 = np.random.rand(self.inputs.shape[1], 6)
        self.weights2 = np.random.rand(6, 1)

        # Output correcto de entrenamiento. y
        self.outputs = outputs

        # Output incorrecto (temporal) ŷ del entrenamiento
        self.output = np.zeros(self.outputs.shape)

        self.error_history = []
        self.epoch_list = []

    # activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.layer1 = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

        # going backwards through the network to update weights
    def backpropagation(self):
        self.error = self.outputs - self.output

        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.outputs - self.output) * self.sigmoid(self.output, deriv=True)))
        d_weights1 = np.dot(self.inputs.T,  (np.dot(2*(self.outputs - self.output) * self.sigmoid(self.output, deriv=True), self.weights2.T) * self.sigmoid(self.layer1, deriv=True)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    # train the neural net for 25,000 iterations

    def train(self, epochs=300):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

            # sself.imprimir(epoch=epoch)

    def imprimir(self, epoch=0):
        print("Iteracion=", epoch, " Error=", self.error_history[-1])
        print("Capa1", self.weights1)
        print("Capa2", self.weights2)

    # function to predict output on new and unseen input data

    def predict(self, new_input):
        intermedio = self.sigmoid(np.dot(new_input, self.weights1))
        prediction = self.sigmoid(np.dot(intermedio, self.weights2))
        return prediction


# create neural network
#NN = NeuralNetwork(inputs, outputs)
NN = NeuralNetwork(inputs_ucg, outputs_ucg)
# train neural network
NN.train(10000)

# create two new examples to predict
input_example = np.array([[1, 0, 0, 1]])
output_example = 1
input_example_2 = np.array([[0, 0, 0, 1]])
output_example_2 = 0

# print the predictions for both examples
print(input_example, NN.predict(input_example), ' - Correct: ', output_example)
print(input_example_2, NN.predict(input_example_2), ' - Correct: ', output_example_2)

# plot the error over the entire training duration
plt.figure(figsize=(15, 5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
