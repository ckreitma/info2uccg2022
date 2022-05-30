# https://www.machinecurve.com/index.php/2021/01/26/creating-a-multilayer-perceptron-with-pytorch-and-lightning/  # pytorch-lightning

import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self):
        super().__init__()
        # Creación de la red neuronal multicapa
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 784),
            nn.ReLU(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


if __name__ == '__main__':

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare MNIST dataset
    dataset_train = MNIST(root='./datasets', download=True, train=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=10, shuffle=True, num_workers=1)

    dataset_test = MNIST(root="./datasets", download=True, train=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 2):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')

    correct = 0
    total = len(dataset_test)

    with torch.no_grad():
        # Iterate through test set minibatchs
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, labels = data

            # Forward pass
            y = mlp.forward(inputs)

            if i % 500 == 499:
                print('Testing: {}'.format(i))
            predictions = torch.argmax(y, dim=1)
            correct += torch.sum((predictions == labels).float())
    print('Test accuracy: {}'.format(correct/total))
