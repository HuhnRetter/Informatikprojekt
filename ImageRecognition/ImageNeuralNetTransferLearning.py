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
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

device = torch.device('dml')

# Parameter
all_classes = ["dog", "flower", "other"]
num_classes = 3
num_epochs = 3
batch_size = 3
learning_rate = 0.01
load_model_param = 1  # 0 == false; 1 == true

learning_rate_string = str(learning_rate).replace('.', '')

MiddleFilename = f"NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = "ImagenetNeuralNet.pth"

writer = SummaryWriter(f'runs/{MiddleFilename}V1')

FILE = f"LetterNeuralNet{MiddleFilename}{EndFilename}"


def outputAcc(n_correct, n_wrong, color):
    acc = 100.0 * n_correct / (n_wrong + n_correct)
    print(f'Accuracy of the network on {color} colors: {acc} % ({n_correct}/{n_wrong + n_correct})\n')


def countPredictedImages(labels, predicted, n_correct_array, n_wrong_array):
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
    floatTensor = floatTensor.view(batch_size)
    # Convert to LongTensor
    longTensor = floatTensor.long()
    return longTensor


def trainingPhase(model, criterion, optimizer, train_loader):
    n_total_steps = len(train_loader)
    n_total_steps_quarter = n_total_steps * .25
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images.to(device)
            labels.to(device)
            model.to(device)
            # Forward pass
            outputs = model(images.to(device))
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
    model.to(device)
    with torch.no_grad():
        print("\n\nStarting with Testing!")
        n_correct_array = [0, 0, 0]
        n_wrong_array = [0, 0, 0]

        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            #labels = convertFloatTensorToLongTensor(labels)
            # max returns (value ,index)
            # predicted = torch.argmax(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            n_correct_array, n_wrong_array = countPredictedImages(labels.to(device), predicted.to(device), n_correct_array, n_wrong_array)

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
    train_dataset = ImageFolder(root='./datasets/train', transform=data_transforms['train'])
    test_dataset = ImageFolder(root='./datasets/test', transform=data_transforms['test'])
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def neuralNetSetup():
    model = models.resnet18(pretrained=True)
    # so not the whole neural net gets rebalanced
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def main():
    model = neuralNetSetup().to(device)
    train_loader, test_loader = dataloaderSetup()
    if load_model_param == 1:
        model = load_model(model)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = trainingPhase(model, criterion, optimizer, train_loader)

    testingPhase(model, test_loader)


if __name__ == "__main__":
    main()
