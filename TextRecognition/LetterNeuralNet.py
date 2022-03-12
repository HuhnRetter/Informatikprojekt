import torch.nn as nn
############## TENSORBOARD ########################
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs
###################################################
from HelperFunctions.HelperPhases import *

# dml only works if there is an amd gpu
device = torch.device("dml")
confusionmatrixdevice = torch.device('cpu')

all_classes = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z"]

# Paths
DATASETPATH = 'C:/Users/matri/Desktop/Informatikprojekt/Backups/BackupNew/TextRecognition/data'
MODELFOLDER = './Models/'
# Parameter
num_epochs = 10
batch_size = 26
learning_rate = 0.001

# 1 == True ; 0 == False
load_model_from_file = 1
save_trained_model = 1

# Automatic Filename for loading and saving
learning_rate_string = str(learning_rate).replace('.', '')
MiddleFilename = f"NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = ".pth"
FILE = f"{MODELFOLDER}LetterNeuralNet{MiddleFilename}{EndFilename}"

# Manuel Filename for loading
FILEPATH = "LetterNeuralNetNE10BS26LR0001.pth"

FILE = f"{MODELFOLDER}{FILEPATH}"
# Writer for Tensorboard
writer = SummaryWriter(f'Tensorboard/runs/{MiddleFilename}')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        # -> n, 1, 28, 28
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 12, 12
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 4, 4
        x = x.view(-1, 16 * 4 * 4)  # -> n, 256
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 26
        return x


def dataloaderSetup():
    """setups both dataloader for the EMNIST datasets

    :return: train_loader and test_loader of the EMNIST dataset
    """
    # EMNIST dataset
    train_dataset = torchvision.datasets.EMNIST(root=DATASETPATH, split='letters', transform=transforms.ToTensor(),
                                                train=True, download=True)
    test_dataset = torchvision.datasets.EMNIST(root=DATASETPATH, split='letters', transform=transforms.ToTensor(),
                                               train=False)
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main():
    model = ConvNet().to(device)
    train_loader, test_loader = dataloaderSetup()
    if load_model_from_file == 1:
        model = load_model(model, FILE)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = trainingPhase(model, criterion, optimizer, train_loader, num_epochs, 0.1, save_trained_model, device,
                              confusionmatrixdevice, writer, FILE, all_classes, 1)

    testingPhase(model, test_loader, writer, FILE, all_classes, 1, confusionmatrixdevice)


if __name__ == "__main__":
    main()
