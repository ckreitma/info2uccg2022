# https://www.kaggle.com/code/sushovansaha9/butterflies-mobilenetv2-pytorch

import time
import torchvision.models as models
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import splitfolders
from torch import nn
import numpy as np
import os
import wget
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

df_train = pd.read_csv('./machine_learning/datasets/mariposas/Butterfly Identification/Training_set.csv')
df_test = pd.read_csv('./machine_learning/datasets/mariposas/Butterfly Identification/Testing_set.csv')
# print(f'train_head={df_train.head()}')
# print(f'test_head={df_test.head()}')
X = df_train.iloc[:, 0]
Y = df_train.iloc[:, 1]
y = list(set(list(Y)))
y.sort()
# print(f'y={y}')
dest = "./machine_learning/datasets/mariposas/BF"

# Descomentar lo siguiente si no existe el directorio BF
# for i in y:
#    os.makedirs(os.path.join(dest, i))
dtdir = './machine_learning/datasets/mariposas/Butterfly Identification/train'
HEIGHT = 224
WIDTH = 224
BS = 256

img = os.path.join(dtdir, X[3])
print(f'Abriendo archivo {img}...')
image = cv2.imread(img)
plt.imshow(image)
plt.axis("off")
image.shape
for i in range(len(X)):
    try:
        destrain = os.path.join(dest, Y[i])
        savepath = os.path.join(destrain, X[i])
        img = os.path.join(dtdir, X[i])
        image = cv2.imread(img)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        cv2.imwrite(savepath, image)
    except:
        print('error')
        pass
splitfolders.ratio('./machine_learning/datasets/mariposas/BF', output="./machine_learning/datasets/mariposas/data", seed=42, ratio=(0.8, 0.2))
# initialize our data augmentation functions
resize = transforms.Resize(size=(WIDTH, HEIGHT))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)
coljtr = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
raf = transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0)
rrsc = transforms.RandomResizedCrop(size=WIDTH, scale=(0.8, 1.0))
ccp = transforms.CenterCrop(size=WIDTH)  # Image net standards
nrml = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([resize, hFlip, vFlip, rotate, coljtr, rrsc, ccp, raf, transforms.ToTensor(), nrml])
valTransforms = transforms.Compose([resize, hFlip, vFlip, rotate, coljtr, rrsc, ccp, raf, transforms.ToTensor(), nrml])

print("[INFO] loading the training and validation dataset...")
trainDataset = ImageFolder(root='./machine_learning/datasets/mariposas/data/train', transform=trainTransforms)
valDataset = ImageFolder(root='./machine_learning/datasets/mariposas/data/val', transform=valTransforms)
print("[INFO] training dataset contains {} samples...".format(len(trainDataset)))
print("[INFO] validation dataset contains {} samples...".format(len(valDataset)))
print("[INFO] creating training and validation set dataloaders...")
trainDataLoader = DataLoader(trainDataset, batch_size=BS, shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size=BS, shuffle=True)
examples = iter(valDataLoader)
example_data, example_targets = examples.next()
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = example_data[i].cpu().numpy().T
    plt.imshow(img)
    plt.axis("off")
plt.show()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Dispositivo={device}')
model = models.mobilenet_v2(pretrained=True)
model.aux_logits = False

# Freeze training for all layers
for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Sequential(
    nn.BatchNorm1d(model.classifier[1].in_features),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(256, len(trainDataset.classes)),
    nn.LogSoftmax(dim=1))
model = model.to(device)

loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(epoch):
    model.train()
    net_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(trainDataLoader):
        (data, target) = (data.to(device), target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        output = torch.exp(output)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss.backward()
        optimizer.step()

        net_loss = net_loss + loss.item()
    acc = correct / len(trainDataLoader.dataset)
    return net_loss, acc


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valDataLoader:
            (data, target) = (data.to(device), target.to(device))
            output = model(data)
            test_loss += loss_func(output, target).item()
            output = torch.exp(output)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(valDataLoader.dataset)
    acc = correct / len(valDataLoader.dataset)
    return test_loss, acc


LOSSTR = []
ACCTE = []
LOSSTE = []
ACCTR = []
n_epochs = 50
for epoch in range(1, n_epochs + 1):
    start = time.time()
    print("--- Epoch {} ---".format(epoch))
    epoch_loss, tracc = train(epoch)
    LOSSTR.append(epoch_loss)
    ACCTR.append(tracc)
    print("\tTrain Accuracy = {} || Train Loss  = {} ".format(tracc, epoch_loss))
    tloss, tacc = test()
    print("\tTest Accuracy =  {} || Test Loss = {} ".format(tacc, tloss))
    ACCTE.append(tacc)
    LOSSTE.append(tloss)
    stop = time.time()
    print("\tTraining time = ", (stop - start))

xx = np.arange(n_epochs)
plt.style.use('fivethirtyeight')
acctr = torch.Tensor(ACCTR).detach().cpu().numpy()
lsstr = torch.Tensor(LOSSTR).detach().cpu().numpy()
accte = torch.Tensor(ACCTE).detach().cpu().numpy()
lsste = torch.Tensor(LOSSTE).detach().cpu().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('ACC vs LOSS')
ax1.plot(xx, acctr, label='Train')
ax1.plot(xx, accte, label='Val')
ax1.legend(loc="best")
ax2.plot(xx, lsstr, label='Train')
ax2.plot(xx, lsste, label='Val')
ax2.legend(loc="best")
plt.show()

testimage = list(paths.list_images('./machine_learning/datasets/mariposas/Butterfly Identification/test'))


def predimg(path):
    from PIL import Image
    image = Image.open(path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    model.eval()
    with torch.no_grad():
        img = load_img(path)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
        img_normalized = transform_norm(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
        img = img_normalized.to(device)
        output = model(img)
        output = torch.exp(output)
        index = output.data.cpu().numpy().argmax()
        result = list(np.around(output.data.cpu().numpy()*100, 1))
        print(result)
        print("PREDICTED CLASS = ", trainDataset.classes[index])


predimg(testimage[5])
predimg(testimage[20])
