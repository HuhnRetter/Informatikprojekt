import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
# tensorboard --logdir=runs
###################################################
from HelperFunctions.HelperFunctions import *

device = torch.device('dml')

all_classes = ["dog", "flower", "other"]

#Paths
DATASETPATH = 'U:/Studium/5.Semester/Informatikprojekt/Backups/BackupNew/ImageRecognition/datasets'
MODELFOLDER = './Models/'
# Parameter
num_classes = len(all_classes)
num_epochs = 3
batch_size = 3
learning_rate = 0.01

# 1 == True ; 0 == False
load_model_from_file = 1

# Automatic Filename for loading and saving
learning_rate_string = str(learning_rate).replace('.', '')
MiddleFilename = f"NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = ".pth"
FILE = f"{MODELFOLDER}ImageTransferLearning{MiddleFilename}{EndFilename}"

# Manuel Filename for loading
#FILE = "ImageTransferLearningNE3BS3LR001ACC94.pth"

# Writer for Tensorboard
writer = SummaryWriter(f'Tensorboard/runs/{MiddleFilename}')



def neuralNetSetup():
    """gets the pretrained model resnet18

    and adds a new Linear layer to it

    :return: returns the pretrained model resnet18
    """
    model = models.resnet18(pretrained=True)
    # so not the whole neural net gets rebalanced
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def dataloaderSetup():
    """setups train and test dataloader from the given path

    :return: returns train and test loader
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
        ]),
    }
    # Image dataset
    train_dataset = ImageFolder(root=f'{DATASETPATH}/train', transform=data_transforms['train'])
    test_dataset = ImageFolder(root=f'{DATASETPATH}/test', transform=data_transforms['test'])
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
    n_total_steps_quarter = n_total_steps * .25
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            model.to(device)
            # Forward pass
            outputs = model(images.to(device))
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
    #device = "cpu"
    model.to(device)
    with torch.no_grad():
        print("\n\nStarting with Testing!")
        n_correct_array = [0, 0, 0]
        n_wrong_array = [0, 0, 0]

        # Console output of the Acc of every class but not as detailed as the confusion matrix
        outputCompleteAcc(n_correct_array, n_wrong_array, test_loader, model, all_classes, 0)

        # Save confusion matrix to Tensorboard
        writer.add_figure(f" testing from: {FILE}", createConfusionMatrix(test_loader, model, all_classes, 0))
        writer.close()


def main():
    model = neuralNetSetup().to(device)
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
