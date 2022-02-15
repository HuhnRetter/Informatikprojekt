import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs
###################################################

from HelperFunctions.HelperFunctions import *

device = torch.device('cpu')

all_classes = ["white", "red", "green", "other"]


#Paths
DATASETFOLDER = './Datasets/'
TRAININGDATASETHLS = f"{DATASETFOLDER}TrainingDatasetHLS.txt"
TESTDATASETHLS = f"{DATASETFOLDER}TestDatasetHLS.txt"
MODELFOLDER = './Models/'
# Parameter
input_size = 2
hidden_size = 25
num_classes = len(all_classes)
num_epochs = 150
batch_size = 4
learning_rate = 0.001

# 1 == True ; 0 == False
load_model_from_file = 1

#Automatic Filename for loading and saving
learning_rate_string = str(learning_rate).replace('.', '')
MiddleFilename = f"HS{hidden_size}NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = "HLS.pth"
FILE = f"{MODELFOLDER}ColorNeuralNet{MiddleFilename}{EndFilename}"


#Manuel Filename for loading
#FILE = "ColorNeuralNetHS25NE1500HLSACC8775.pth"
#FILE = "ColorNeuralNetHS25NE10BS10HLSACC9075.pth"

#Writer for Tensorboard
writer = SummaryWriter(f'Tensorboard/runs/{MiddleFilename}')



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """initializes NeuralNet

        :param input_size: input_size of the first linear layer
        :param hidden_size: input_size of the second linear layer
        :param num_classes: number of total classes of type int
        """
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """forwards values through neural net

        :param x: value that will be forwarded
        :return: returns modified value after passing every layer
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


class ColorDataset(Dataset):
    """color dataset from .txt File

    example of dataset: class_id,h,l --> 0,0.08333333333333333,0.9137254901960784

    Attributes:

    - n_samples --> number of rows/samples in dataset
    - n_x_data --> hsl values of type ndarray
    - n_y_data --> labels of type ndarray
    - n_transform --> transform function (for example ToTensor)
    """
    def __init__(self, DATASETPATH, transform=None):
        """initializes ColorDataset

        :param DATASETPATH: datasetpath of type String
        :param transform: transform function (for example ToTensor)
        """
        # read with numpy
        xy = np.loadtxt(DATASETPATH, delimiter=',', dtype=np.float32)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        """get label and features from given index

        if ColorDataset has a transform function the data will be transformed before returning

        :param index: index of label and features of type int
        :return: returns label and features as combined variable
        """
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    """Convert ndarrays to Tensors

    """
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    """multiply inputs with a given factor

    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


def dataloaderSetup(DATASETPATH, normalized):
    """setups the data_loader for the given DATASETPATH

    :param DATASETPATH: String with the path to the dataset
    :param normalized: Int acts as a boolean to indicate if the dataset is normalized (1 == True, 0 == False) --> only works for BGR values NOT for HSL
    :return: returns the data_loader
    """
    if normalized == 1:
        composed = torchvision.transforms.Compose([ToTensor()])
    else:
        composed = torchvision.transforms.Compose([ToTensor(), MulTransform(1 / 255)])
    dataset = ColorDataset(DATASETPATH, transform=composed)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def trainingPhase(model, criterion, optimizer, train_loader):
    """trains the given model with the given parameters.

    iterates once through the train_loader in each epoch
    and updates the weights in the model

    After every quarter step update the acc and loss graph in Tensorboard
    and after every epoch create Confusion matrix

    :param model: current model of the class NeuralNet
    :param criterion: loss function from torch.nn.modules.loss (for Example CrossEntropyLoss)
    :param optimizer: optimizer from torch.optim (for Example Adam)
    :param train_loader: dataloader with Training dataset
    :return: returns trained model of the class NeuralNet
    """
    n_total_steps = len(train_loader)
    n_total_steps_quarter = n_total_steps*.25
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (hsl, labels) in enumerate(train_loader):
            model.to(device)
            # Forward pass
            outputs = model(hsl.to(device))
            labels = convertFloatTensorToLongTensor(labels)
            loss = criterion(outputs.to(device), labels.to(device))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            if (i + 1) % (n_total_steps_quarter) == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
                ############## TENSORBOARD ########################
                drawGraph(writer, running_loss, running_correct, n_total_steps, n_total_steps_quarter, epoch, i, predicted)
                running_correct = 0
                running_loss = 0.0
                ###################################################
        # Save confusion matrix to Tensorboard
        writer.add_figure(f"Confusion matrix training from: {FILE}", createConfusionMatrix(train_loader, model, all_classes, 0), epoch)
        writer.close()

    model.to(torch.device("cpu"))
    torch.save(model.state_dict(), FILE)
    return model


def testingPhase(model, test_loader):
    """tests the model

    outputs Acc of all classes and creates a Confusionmatrix

    :param model: current model of the class NeuralNet
    :param test_loader: dataloader with Test dataset
    """
    with torch.no_grad():
        print("\n\nStarting with Testing!")
        n_correct_array = [0, 0, 0, 0]
        n_wrong_array = [0, 0, 0, 0]

        # Console output of the Acc of every class but not as detailed as the confusion matrix
        outputCompleteAcc(n_correct_array, n_wrong_array, test_loader, model, all_classes, 0)

        # Save confusion matrix to Tensorboard
        writer.add_figure(f"Confusion matrix testing from: {FILE}", createConfusionMatrix(test_loader, model, all_classes, 0))
        writer.close()



def main():
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    if load_model_from_file == 1:
        model = load_model(model, FILE)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = dataloaderSetup(TRAININGDATASETHLS, 1)
        model = trainingPhase(model, criterion, optimizer, train_loader)

    test_loader = dataloaderSetup(TESTDATASETHLS, 1)
    testingPhase(model, test_loader)


if __name__ == "__main__":
    main()
