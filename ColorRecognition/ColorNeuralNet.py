import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs
###################################################
from HelperFunctions.HelperPhases import *

device = torch.device('cpu')
confusionmatrixdevice = torch.device('cpu')

all_classes = ["white", "red", "green", "other"]

# Paths
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
save_trained_model = 1

# Automatic Filename for loading and saving
learning_rate_string = str(learning_rate).replace('.', '')
MiddleFilename = f"HS{hidden_size}NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = "HLS.pth"
FILE = f"{MODELFOLDER}ColorNeuralNet{MiddleFilename}{EndFilename}"

# Manuel Filename for loading
# FILE = "ColorNeuralNetHS25NE1500HLSACC8775.pth"
# FILE = "ColorNeuralNetHS25NE10BS10HLSACC9075.pth"

# Writer for Tensorboard
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


def main():
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    if load_model_from_file == 1:
        model = load_model(model, FILE)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = dataloaderSetup(TRAININGDATASETHLS, 1)
        model = trainingPhase(model, criterion, optimizer, train_loader, num_epochs, 0.1, save_trained_model, device,
                              confusionmatrixdevice, writer, FILE, all_classes, 0)

    test_loader = dataloaderSetup(TESTDATASETHLS, 1)
    testingPhase(model, test_loader, writer, FILE, all_classes, 0, confusionmatrixdevice)


if __name__ == "__main__":
    main()
