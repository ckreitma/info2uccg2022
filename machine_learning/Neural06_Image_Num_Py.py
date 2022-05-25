import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# pip install pillow
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')


mnist_train = datasets.MNIST(
    root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(
    root="./datasets", train=False, transform=transforms.ToTensor(), download=True)


# Pick out the 4th (0-indexed) example from the training set
image, label = mnist_train[1811]

print(f'Cantidad de entrenamiento={len(mnist_train)}')
arr = np.array(image)
print(f'Shape de arr={arr.shape}')
print(f'arr[0][3]=\n{arr[0][3]}')

# Plot the image
#print("Default image shape: {}".format(image.shape))
image = image.reshape([28, 28])
#print("Reshaped image shape: {}".format(image.shape))
plt.imshow(image, cmap="gray")

# Print the label
print(f'The label for this image: {label}')

# https://stackoverflow.com/questions/42812230/why-plt-imshow-doesnt-display-the-image
plt.show()
