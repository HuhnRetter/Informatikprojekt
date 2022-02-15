import torch.nn as nn
############## TENSORBOARD ########################
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs
###################################################
from HelperFunctions.HelperFunctions import *

#dml only works if there is an amd gpu
device = torch.device("dml")

all_classes = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z"]

#Paths
DATASETPATH = 'U:/Studium/5.Semester/Informatikprojekt/Backups/BackupNew/TextRecognition/data'
MODELFOLDER = './Models/'
# Parameter
num_epochs = 10
batch_size = 26
learning_rate = 0.001

# 1 == True ; 0 == False
load_model_from_file = 1

#Automatic Filename for loading and saving
learning_rate_string = str(learning_rate).replace('.', '')
MiddleFilename = f"NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = ".pth"
FILE = f"{MODELFOLDER}LetterNeuralNet{MiddleFilename}{EndFilename}"


#Manuel Filename for loading
FILEPATH = "LetterNeuralNetNE10BS26LR0001.pth"

FILE = f"{MODELFOLDER}{FILEPATH}"
#Writer for Tensorboard
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
        x = x.view(-1, 16 * 4 * 4)           # -> n, 256
        x = F.relu(self.fc1(x))              # -> n, 120
        x = F.relu(self.fc2(x))              # -> n, 84
        x = self.fc3(x)                      # -> n, 26
        return x


def dataloaderSetup():
    """setups both dataloader for the EMNIST datasets

    :return: train_loader and test_loader of the EMNIST dataset
    """
    # EMNIST dataset
    train_dataset = torchvision.datasets.EMNIST(root=DATASETPATH,  split='letters', transform=transforms.ToTensor(), train=True, download=True)
    test_dataset = torchvision.datasets.EMNIST(root=DATASETPATH, split='letters', transform=transforms.ToTensor(), train=False)
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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
    n_total_steps_quarter = n_total_steps * .1
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (letters, labels) in enumerate(train_loader):
            #labels transforms because labels start with 1
            labels = torch.add(labels, -1)
            model.to(device)
            # Forward pass
            outputs = model(letters.to(device))
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
        writer.add_figure(f"Confusion matrix training from: {FILE}", createConfusionMatrix(train_loader, model, all_classes, 1), epoch)
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
        n_correct_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        n_wrong_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Console output of the Acc of every class but not as detailed as the confusion matrix
        outputCompleteAcc(n_correct_array, n_wrong_array, test_loader, model, all_classes, 1)

        # Save confusion matrix to Tensorboard
        writer.add_figure(f"Confusion matrix testing from: {FILE}", createConfusionMatrix(test_loader, model, all_classes, 1))
        writer.close()



def main():
    model = ConvNet().to(device)
    train_loader, test_loader = dataloaderSetup()
    if load_model_from_file == 1:
        model = load_model(model, FILE)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = trainingPhase(model, criterion, optimizer, train_loader)

    testingPhase(model, test_loader)

if __name__ == "__main__":
    main()
