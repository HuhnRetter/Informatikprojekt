import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt


def createConfusionMatrix(loader, model, all_classes, label_start_at):
    """creates Confusionmatrix from given Dataloader and given Model

    :param loader: An instance of the class ColorDataset
    :param model: current model of the class NeuralNet
    :param all_classes: list of all possible classes
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :return: returns confusion matrix as figure
    """
    y_pred = []  # save prediction
    y_true = []  # save ground truth
    # iterate over data
    n_total_steps = len(loader)

    for i, (inputs, labels) in enumerate(loader):
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

        y_pred.extend(output)  # save prediction

        if (label_start_at == 1):
            # labels transforms because labels start with 1
            labels = torch.add(labels, -1)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth
        # print to see progress because sometimes it takes a while and pauses
        # print(f'Step [{i + 1}/{n_total_steps}]')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(all_classes), index=[i for i in all_classes],
                         columns=[i for i in all_classes])
    plt.figure(figsize=(21, 7))
    return sn.heatmap(df_cm.round(4), annot=True).get_figure()

def drawGraph(writer, running_loss, running_correct, n_total_steps, n_total_steps_quarter, epoch, i, predicted):
    """

    Args:
        writer:
        running_loss:
        running_correct:
        n_total_steps:
        n_total_steps_quarter:
        epoch:
        i:
        predicted:
    """
    writer.add_scalar('training loss', running_loss / n_total_steps_quarter, epoch * n_total_steps + i)
    running_accuracy = running_correct / n_total_steps_quarter / predicted.size(0)
    writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
    writer.close()


def calculateAcc(n_correct, n_wrong, class_name):
    """calculates the accuracy of guessing the class right from the given arrays

    Args:
        n_correct: number of correct guesses as int
        n_wrong: number of wrong guesses as int
        class_name: name of the individual guessed class
    """
    acc = 100.0 * n_correct / (n_wrong + n_correct)
    print(f'Accuracy of the network on {class_name}: {acc} % ({n_correct}/{n_wrong + n_correct})\n')


def outputCompleteAcc(n_correct_array, n_wrong_array, test_loader, model, all_classes, label_start_at):
    """outputs the acc of every class as well as the average acc of all classes

    outputs the same as the confusion matrix, so it is only used if the content should be seen in the console

    Args:
        n_correct_array: list with size of classes for example:  at index 0 is class number 0 and it holds the value of the times it was guessed correct
        n_wrong_array: list with size of classes for example:  at index 0 is class number 0 and it holds the value of the times it was guessed wrong
        test_loader: dataloader with Test dataset
        model: current model of the class NeuralNet
        all_classes: list of all class names
        label_start_at: value for correction of all labels in the dataset if starts at 1
    """
    for inputs, labels in test_loader:
        outputs = model(inputs)
        labels = convertFloatTensorToLongTensor(labels)

        if(label_start_at == 1):
            # labels transforms because labels start with 1
            labels = torch.add(labels, -1)

        _, predicted = torch.max(outputs.data, 1)
        n_correct_array, n_wrong_array = countCorrectlyGuessed(labels, predicted, n_correct_array, n_wrong_array)

    n_correct = 0
    n_wrong = 0
    for counter, color in enumerate(all_classes):
        calculateAcc(n_correct_array[counter], n_wrong_array[counter], color)
        n_correct += n_correct_array[counter]
        n_wrong += n_wrong_array[counter]
    calculateAcc(n_correct, n_wrong, "all")


def countCorrectlyGuessed(labels, predicted, n_correct_array, n_wrong_array):
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


def load_model(model, FILE):
    """loads an initialized model from a given path

    :param model: initialized model of type ColorNeuralNet.NeuralNet
    :return: returns the fully loaded model
    """
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load(FILE))
    model.eval()
    return model
