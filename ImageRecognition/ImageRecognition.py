import math
import os

import PIL.ImageOps
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image

import ImageNeuralNetTransferLearning
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Parameters
TESTFOLDER = './testImages'
FILENEURALNET = "LetterNeuralNetNE3BS3LR001ImagenetNeuralNetACC94.pth"
all_classes = ["dog", "flower", "other"]

#batch_size must be same as images
batch_size = 9


def setupDatasetLoader():
    multitransform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5, 0.5),
    ])
    test_dataset = ImageFolder(root=TESTFOLDER, transform=multitransform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return test_loader


def setupModel():
    model = ImageNeuralNetTransferLearning.neuralNetSetup()
    model.load_state_dict(torch.load(FILENEURALNET))
    model.eval()
    return model


def setup():
    batch_size_sqrt = int(math.sqrt(batch_size))
    fig, ax = plt.subplots(batch_size_sqrt, batch_size_sqrt)
    ax = ax.ravel()
    return ax, fig


def useImageRecognition(test_loader, model, ax, fig):

    with torch.no_grad():
        i = 0
        for inputs, labels in test_loader:
            for image in inputs:
                #print(image)
                np_array = image.numpy()
                np_array = np_array.swapaxes(0, 2)
                #print(np_array)
                ax[i].imshow(np_array)
                i += 1
            i = 0
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for batchNumber in range(predicted.size(dim=0)):
                ax[i].set_title(all_classes[predicted[batchNumber]])
                i += 1
            fig.tight_layout()
            plt.show()

if __name__ == "__main__":
    test_loader = setupDatasetLoader()
    model = setupModel()
    ax, fig = setup()
    useImageRecognition(test_loader, model, ax, fig)
