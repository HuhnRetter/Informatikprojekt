import math

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np

import ImageTransferLearning

# Parameters
TESTFOLDER = './TestImages'
MODELFOLDER = './Models/'
MODELFILE = "ImageTransferLearningNE3BS3LR001ACC94.pth"
FILENEURALNET = MODELFOLDER + MODELFILE
all_classes = ["dog", "flower", "other"]

# batch_size must be same as images
batch_size = 9


def setupDatasetLoader():
    """transforms the test images

    :return: returns the data_loader for the test images
    """
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
    """loads the model from the given path

    :return: returns the loaded model
    """
    model = ImageTransferLearning.neuralNetSetup()
    model.load_state_dict(torch.load(FILENEURALNET))
    model.eval()
    return model


def setup():
    """setups the output format of the test images

    :return: returns ax for adding images into the plot and fig for defining the layout
    """
    batch_size_sqrt = int(math.sqrt(batch_size))
    fig, ax = plt.subplots(batch_size_sqrt, batch_size_sqrt)
    ax = ax.ravel()
    return ax, fig


def useImageRecognition(test_loader, model, ax, fig):
    """guesses the classes for the given images

    :param test_loader: dataloader for the test images
    :param model: trained model for guessing the classes
    :param ax: array for the images of type ndarray
    :param fig: for defining the layout for example  fig.tight_layout
    """
    with torch.no_grad():
        i = 0
        for inputs, labels in test_loader:
            for image in inputs:
                np_array = image.numpy()
                np_array = np_array.swapaxes(0, 2)
                np_array = np.rot90(np_array, 3)
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
