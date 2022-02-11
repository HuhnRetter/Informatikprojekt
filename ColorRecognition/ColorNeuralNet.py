import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs
###################################################

############## Confusionmatrix ########################
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
###################################################

device = torch.device('cpu')

all_classes = ["white", "red", "green", "other"]

# Parameter
input_size = 2
hidden_size = 25
num_classes = len(all_classes)
num_epochs = 150
batch_size = 4
learning_rate = 0.001

# 1 == True ; 0 == False
load_model_from_file = 0

#Automatic Filename for loading and saving
learning_rate_string = str(learning_rate).replace('.', '')
MiddleFilename = f"HS{hidden_size}NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = "HLS.pth"
FILE = f"ColorNeuralNet{MiddleFilename}{EndFilename}"


#Manuel Filename for loading
#FILE = "ColorNeuralNetHS25NE1500HLSACC8775.pth"
#FILE = "ColorNeuralNetHS25NE10BS10HLSACC9075.pth"

#Writer for Tensorboard
writer = SummaryWriter(f'runs/{MiddleFilename}')

#Paths for Datasets
TRAININGDATASETHLS = "TrainingDatasetHLS.txt"
TESTDATASETHLS = "TestDatasetHLS.txt"


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
    for inputs, labels in loader:
        output = model(inputs.to(device))  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * len(all_classes), index=[i for i in all_classes],
                         columns=[i for i in all_classes])
    plt.figure(figsize=(21, 7))
    return sn.heatmap(df_cm.round(4), annot=True).get_figure()


def outputAcc(n_correct, n_wrong, color):
    acc = 100.0 * n_correct / (n_wrong + n_correct)
    print(f'Accuracy of the network on {color} colors: {acc} % ({n_correct}/{n_wrong + n_correct})\n')


def countPredictedColors(labels, predicted, n_correct_array, n_wrong_array):
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

    :param model: initialized model of type ColorNeuralNet.NeuralNet
    :return: returns the fully loaded model
    """
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load(FILE))
    model.eval()
    return model


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

            outputs = model(hsl.to(device))

            labels = convertFloatTensorToLongTensor(labels)
            loss = criterion(outputs.to(device), labels.to(device))

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
    with torch.no_grad():
        print("\n\nStarting with Testing!")
        n_correct_array = [0, 0, 0, 0]
        n_wrong_array = [0, 0, 0, 0]

        for hsl, labels in test_loader:
            outputs = model(hsl.to(device))
            labels = convertFloatTensorToLongTensor(labels)
            labels.to(device)

            _, predicted = torch.max(outputs.data, 1)
            n_correct_array, n_wrong_array = countPredictedColors(labels, predicted, n_correct_array, n_wrong_array)

        # Save confusion matrix to Tensorboard
        writer.add_figure(f"Confusion matrix testing from: {FILE}", createConfusionMatrix(test_loader, model))
        writer.close()

        counter = 0
        n_correct = 0
        n_wrong = 0
        for color in all_classes:
            outputAcc(n_correct_array[counter], n_wrong_array[counter], color)
            n_correct += n_correct_array[counter]
            n_wrong += n_wrong_array[counter]
            counter += 1
        outputAcc(n_correct, n_wrong, "all")



def main():
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    train_loader = dataloaderSetup(TRAININGDATASETHLS, 1)
    if load_model_from_file == 1:
        model = load_model(model)
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
