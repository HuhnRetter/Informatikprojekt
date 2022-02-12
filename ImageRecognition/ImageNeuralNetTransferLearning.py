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

############## Confusionmatrix ########################
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
###################################################
device = torch.device('dml')

all_classes = ["dog", "flower", "other"]

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
EndFilename = "ImagenetNeuralNet.pth"
FILE = f"ImageTransferLearning{MiddleFilename}{EndFilename}"

# Manuel Filename for loading
FILE = "LetterNeuralNetNE3BS3LR001ImagenetNeuralNetACC94.pth"

# Writer for Tensorboard
writer = SummaryWriter(f'runs/{MiddleFilename}')


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

def createConfusionMatrix(loader, model):
    """creates Confusionmatrix from given Dataloader and given Model

    :param loader: An instance of the class ColorDataset
    :param model: current model of the class NeuralNet
    :return: returns confusion matrix as figure
    """
    y_pred = [] # save prediction
    y_true = [] # save ground truth
    model.to(device)
    # iterate over data
    n_total_steps = len(loader)
    n_total_steps_quarter = n_total_steps * .1

    for i, (inputs, labels) in enumerate(loader):
        output = model(inputs.to(device))  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

        print(f'Step [{i + 1}/{n_total_steps}]')


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * len(all_classes), index=[i for i in all_classes],
                         columns=[i for i in all_classes])
    plt.figure(figsize=(21, 7))
    return sn.heatmap(df_cm.round(4), annot=True).get_figure()

def outputAcc(n_correct, n_wrong, image):
    acc = 100.0 * n_correct / (n_wrong + n_correct)
    print(f'Accuracy of the network on {image} images: {acc} % ({n_correct}/{n_wrong + n_correct})\n')


def countPredictedImages(labels, predicted, n_correct_array, n_wrong_array):
    """compares the two tensors labels and predicted.
    Counts how many elements of the two given tensors are the same

    :param labels: tensor with class_id as its elements
    :param predicted: tensor with class_id as its elements (return value of model)
    :param n_correct_array: List[int] acts as counter for every right guess for each class
    :param n_wrong_array: List[int] acts as counter for every wrong guess for each class
    :return: returns updated n_correct_array and n_wrong_array
    """

    for batchNumber in range(predicted.size(dim=0)):
        if labels[batchNumber] == predicted[batchNumber]:
            n_correct_array[predicted[batchNumber]] += 1
        else:
            n_wrong_array[labels[batchNumber]] += 1
            print(f"predicted: {predicted}")
            print(f"labels: {labels}")
            print("Wrong\n")

    return n_correct_array, n_wrong_array


def convertFloatTensorToLongTensor(floatTensor):
    """converts a float tensor to a long tensor

    :param floatTensor: tensor of type float
    :return: returns a tensor of type long
    """
    # Axis correction
    floatTensor = floatTensor.view(floatTensor.size(dim=0))
    # Convert to LongTensor
    longTensor = floatTensor.long()
    return longTensor


def load_model(model):
    """loads an initialized model from a given path

        :param model: initialized model
        :return: returns the fully loaded model
        """
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load(FILE))
    model.eval()
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
    train_dataset = ImageFolder(root='./datasets/train', transform=data_transforms['train'])
    test_dataset = ImageFolder(root='./datasets/test', transform=data_transforms['test'])
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
            images.to(device)
            labels.to(device)
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
                writer.add_scalar('training loss', running_loss / n_total_steps_quarter, epoch * n_total_steps + i)
                running_accuracy = running_correct / n_total_steps_quarter / predicted.size(0)
                writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
                running_correct = 0
                running_loss = 0.0
                writer.close()
                ###################################################
        # Save confusion matrix to Tensorboard
        writer.add_figure(f"Confusion matrix training from: {FILE}", createConfusionMatrix(train_loader, model), epoch)
        writer.close()
    model.to(torch.device("cpu"))
    torch.save(model.state_dict(), FILE)
    return model


def testingPhase(model, test_loader):
    """tests the model

    iterates once through the test_loader
    and checks how many classes are correctly guessed

    after that creates a Confusionmatrix visible in Tensorboard

    :param model: current model of the class NeuralNet
    :param test_loader: dataloader with Test dataset
    """
    #device = "cpu"
    model.to(device)
    with torch.no_grad():
        print("\n\nStarting with Testing!")
        n_correct_array = [0, 0, 0]
        n_wrong_array = [0, 0, 0]

        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            n_correct_array, n_wrong_array = countPredictedImages(labels.to(device), predicted.to(device), n_correct_array, n_wrong_array)

        # Save confusion matrix to Tensorboard
        writer.add_figure(f" testing from: {FILE}", createConfusionMatrix(test_loader, model))
        writer.close()

        counter = 0
        n_correct = 0
        n_wrong = 0
        for letter in all_classes:
            outputAcc(n_correct_array[counter], n_wrong_array[counter], letter)
            n_correct += n_correct_array[counter]
            n_wrong += n_wrong_array[counter]
            counter += 1
        outputAcc(n_correct, n_wrong, "all")


def main():
    model = neuralNetSetup().to(device)
    train_loader, test_loader = dataloaderSetup()
    if load_model_from_file == 1:
        model = load_model(model)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = trainingPhase(model, criterion, optimizer, train_loader)

    testingPhase(model, test_loader)


if __name__ == "__main__":
    main()
