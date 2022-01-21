import cv2
import numpy as np
import random as r
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here

###################################################


device = torch.device('cpu')

# Parameter
input_size = 2
hidden_size = 4
num_classes = 4

num_epochs = 35
batch_size = 10
learning_rate = 0.01  # Möglicherweise Essentiell für die Erkennung

MiddleFilename = f"HS{hidden_size}NE{num_epochs}BS{batch_size}"
EndFilename = "HLS.pth"

writer = SummaryWriter(f'runs/{MiddleFilename}V3')

FILE = f"ColorNeuralNet{MiddleFilename}{EndFilename}"
TRAININGDATASETBGR = "TrainingDatasetBGR.txt"
TESTDATASETBGR = "TestDatasetBGR.txt"

TRAININGDATASETHLS = "TrainingDatasetHLS.txt"
TESTDATASETHLS = "TestDatasetHLS.txt"



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        # out = self.tanh(out)
        out = self.l2(out)

        # no activation and no softmax at the end
        return out


class ColorDataset(Dataset):

    def __init__(self, DATASETPATH, transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt(DATASETPATH, delimiter=',', dtype=np.float32)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


def outputAcc(n_correct, n_wrong, color):
    acc = 100.0 * n_correct / (n_wrong + n_correct)
    print(f'Accuracy of the network on {color} colors: {acc} % ({n_correct}/{n_wrong + n_correct})\n')


def countPredictedColors(labels, predicted, n_correct_array, n_wrong_array):
    for batchNumber in range(batch_size):
        if labels[batchNumber] == predicted[batchNumber]:
            n_correct_array[predicted[batchNumber]] += 1
            print("Correct\n")
        else:
            n_wrong_array[labels[batchNumber]] += 1
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
    n_total_steps_quarter = n_total_steps*.25
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (bgr, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(bgr)
            # print(f"outputs: {outputs}")
            # print(f"labels: {labels}")
            labels = convertFloatTensorToLongTensor(labels)
            # print(f"labels after: {labels}")
            loss = criterion(outputs, labels)
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
    torch.save(model.state_dict(), FILE)
    return model


def testingPhase(model, test_loader):
    with torch.no_grad():
        print("\n\nStarting with Testing!")
        n_correct_array = [0, 0, 0, 0]
        n_wrong_array = [0, 0, 0, 0]

        for bgr, labels in test_loader:
            outputs = model(bgr)
            labels = convertFloatTensorToLongTensor(labels)
            # max returns (value ,index)
            # predicted = torch.argmax(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            print(f"predicted: {predicted}")
            print(f"labels: {labels}")
            n_correct_array, n_wrong_array = countPredictedColors(labels, predicted, n_correct_array, n_wrong_array)

        outputAcc(n_correct_array[0], n_wrong_array[0], "white")
        outputAcc(n_correct_array[1], n_wrong_array[1], "red")
        outputAcc(n_correct_array[2], n_wrong_array[2], "green")
        outputAcc(n_correct_array[3], n_wrong_array[3], "other")

        n_correct = n_correct_array[0] + n_correct_array[1] + n_correct_array[2] + n_correct_array[3]
        n_wrong = n_wrong_array[0] + n_wrong_array[1] + n_wrong_array[2] + n_wrong_array[3]
        outputAcc(n_correct, n_wrong, "all")


def load_model(model):
    # loading existing model
    model.load_state_dict(torch.load(FILE))
    model.eval()
    return model


def dataloaderSetup(DATASETPATH, normalized):
    # dataset = ColorBGRDataset()
    if normalized == 1:
        composed = torchvision.transforms.Compose([ToTensor()])
    else:
        composed = torchvision.transforms.Compose([ToTensor(), MulTransform(1 / 255)])
    dataset = ColorDataset(DATASETPATH, transform=composed)

    first_data = dataset[0]
    features, labels = first_data
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    return train_loader


def main(load_model):
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    if load_model == 1:
        model = load_model(model)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        train_loader = dataloaderSetup(TRAININGDATASETHLS, 1)
        model = trainingPhase(model, criterion, optimizer, train_loader)

    test_loader = dataloaderSetup(TESTDATASETHLS, 1)
    testingPhase(model, test_loader)


if __name__ == "__main__":
    main(0)
