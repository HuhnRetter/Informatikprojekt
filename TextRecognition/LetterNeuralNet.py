import cv2
import numpy as np
import random as r
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
# tensorboard --logdir=runs
###################################################


#device = torch.device('cpu')
#device = "dml"
device = torch.device("dml")

# Parameter
all_classes = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z"]
num_epochs = 5
batch_size = 66
learning_rate = 0.001  # Möglicherweise Essentiell für die Erkennung


learning_rate_string = str(learning_rate).replace('.', '')

MiddleFilename = f"NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = "LetterNeuralNet.pth"

writer = SummaryWriter(f'runs/{MiddleFilename}V2')

#FILE = f"LetterNeuralNet{MiddleFilename}{EndFilename}"
FILE = "LetterNeuralNetNE100BS100LR0001LetterNeuralNet.pth"

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

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 26)

    def forward(self, x):
        # -> n, 1, 28, 28
        x = self.pool(F.relu(self.conv1(x))) # -> n, 6, 12, 12
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 4, 4
        x = x.view(-1, 16 * 4 * 4)          # -> n, 256
        x = F.relu(self.fc1(x))               # -> n, 128
        x = F.relu(self.fc2(x))              # -> n, 64
        x = self.fc3(x)                    # -> n, 26
        return x


def outputAcc(n_correct, n_wrong, color):
    acc = 100.0 * n_correct / (n_wrong + n_correct)
    print(f'Accuracy of the network on {color} colors: {acc} % ({n_correct}/{n_wrong + n_correct})\n')


def countPredictedLetters(labels, predicted, n_correct_array, n_wrong_array):
    for batchNumber in range(predicted.size(dim=0)):
        if labels[batchNumber] == predicted[batchNumber]:
            n_correct_array[predicted[batchNumber]] += 1
            #print("Correct\n")
        else:
            n_wrong_array[labels[batchNumber]] += 1
            print(f"predicted: {predicted}")
            print(f"labels: {labels}")
            print("Wrong\n")

    return n_correct_array, n_wrong_array


def convertFloatTensorToLongTensor(floatTensor):
    # Axis correction
    floatTensor = floatTensor.view(floatTensor.size(dim=0))
    # Convert to LongTensor
    longTensor = floatTensor.long()
    return longTensor


def trainingPhase(model, criterion, optimizer, train_loader):
    n_total_steps = len(train_loader)
    n_total_steps_quarter = n_total_steps * .1
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (letters, labels) in enumerate(train_loader):
            #print(labels)
            #labels transforms because labels start with 1
            labels = torch.add(labels, -1)
            letters.to(device)
            labels.to(device)
            model.to(device)
            #print(labels)
            # Forward pass
            outputs = model(letters.to(device))
            # print(f"outputs: {outputs}")
            # print(f"labels: {labels}")
            labels = convertFloatTensorToLongTensor(labels)
            # print(f"labels after: {labels}")
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
                writer.add_scalar('training loss', running_loss / n_total_steps_quarter, epoch * n_total_steps + i)
                running_accuracy = running_correct / n_total_steps_quarter / predicted.size(0)
                writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
                running_correct = 0
                running_loss = 0.0
                writer.close()
                ###################################################
    model.to(torch.device("cpu"))
    torch.save(model.state_dict(), FILE)
    return model


def testingPhase(model, test_loader):
    with torch.no_grad():
        print("\n\nStarting with Testing!")
        n_correct_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        n_wrong_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for inputs, labels in test_loader:
            print(inputs.type())
            print(inputs[0])
            print(inputs[0].dtype)
            inputs.to(device)
            outputs = model(inputs)
            labels = convertFloatTensorToLongTensor(labels)
            # labels transforms because labels start with 1
            labels = torch.add(labels, -1)
            labels.to(device)
            # max returns (value ,index)
            # predicted = torch.argmax(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            n_correct_array, n_wrong_array = countPredictedLetters(labels, predicted, n_correct_array, n_wrong_array)

        counter = 0
        n_correct = 0
        n_wrong = 0
        for letter in all_classes:
            outputAcc(n_correct_array[counter], n_wrong_array[counter], letter)
            n_correct += n_correct_array[counter]
            n_wrong += n_wrong_array[counter]
            counter += 1
        outputAcc(n_correct, n_wrong, "all")


def load_model(model):
    # loading existing model
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load(FILE))
    model.eval()
    return model


def dataloaderSetup():
    # EMNIST dataset
    train_dataset = torchvision.datasets.EMNIST(root='./data',  split='letters', transform=transforms.ToTensor(), train=True, download=True)
    test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', transform=transforms.ToTensor() ,train=False)
    temp_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', transform=transforms.ToTensor() ,train=True)
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    temp_loader = torch.utils.data.DataLoader(dataset=temp_dataset, batch_size=batch_size, shuffle=True)

    for temp, labels in temp_loader:
        print(temp[0])
        print(all_classes[labels[0-1]])
        print("\n\n\n")
    sys.exit()

    return train_loader, test_loader



def main(load_model_parameter):
    model = ConvNet().to(device)
    train_loader, test_loader = dataloaderSetup()
    if load_model_parameter == 1:
        model = load_model(model)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = trainingPhase(model, criterion, optimizer, train_loader)

    testingPhase(model, test_loader)


if __name__ == "__main__":
    main(1)
