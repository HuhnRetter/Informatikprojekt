import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
import sys
import time


def createConfusionMatrix(loader, model, all_classes, label_start_at, output_examples_check, device):
    """creates Confusionmatrix from given Dataloader and given Model & gives out a console output

    :param loader: An instance of the class ColorDataset
    :param model: current model of the class NeuralNet
    :param all_classes: list of all possible classes
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :param output_examples_check: value to enable the output of wrong examples 1 --> True; 0 --> False
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :return: returns confusion matrix as figure
    """
    n_correct_array = list()
    n_wrong_array = list()
    output_examples = list()

    for i in range(len(all_classes)):
        n_correct_array.append(0)
        n_wrong_array.append(0)

    y_pred = []  # save prediction
    y_true = []  # save ground truth
    # iterate over data

    starting_time = time.time()

    print("\nCreating Confusion Matrix ...")

    n_total_steps = len(loader)

    model.to(device)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            output = model(inputs.to(device))  # Feed Network

            # for output in console
            _, predicted = torch.max(output.data, 1)

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

            y_pred.extend(output)  # save prediction

            if label_start_at == 1:
                # Label transforms because labels start with 1
                labels = torch.add(labels, -1)

            longLabel = convertFloatTensorToLongTensor(labels)
            # for output in console
            n_correct_array, n_wrong_array, output_examples = countCorrectlyGuessed(longLabel.to(device), predicted,
                                                                                    n_correct_array, n_wrong_array,
                                                                                    output_examples)

            labels = labels.data.cpu().numpy()

            y_true.extend(labels)  # save ground truth

            # print to see progress because sometimes it takes a while and pauses
            progressBarWithTime(i, n_total_steps, starting_time)
    # Print accuracy in console
    print("\n\n")
    if output_examples_check == 1:
        print("###########################Examples for wrong predictions##################################\n")
        for example in output_examples:
            print(example)

    n_correct = 0
    n_wrong = 0
    print("\n#####################################Statistics############################################\n")
    for counter, color in enumerate(all_classes):
        calculateAcc(n_correct_array[counter], n_wrong_array[counter], color)
        n_correct += n_correct_array[counter]
        n_wrong += n_wrong_array[counter]
    calculateAcc(n_correct, n_wrong, "all")
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(all_classes), index=[i for i in all_classes],
                         columns=[i for i in all_classes])

    plt.figure(figsize=(21, 7))
    return sn.heatmap(df_cm.round(4), annot=True).get_figure()


def calculateRemainingTime(starting_time, current_time, i, n_total_steps):
    """calculated the remaining time with the help of the given parameters

    :param starting_time: starting time of the progressbar
    :param current_time: current time as the calculations take place
    :param i: current step as int
    :param n_total_steps: total steps/batches of the epoch as int

    Returns: returns approximated remaining time in seconds

    """
    if i <= 0:
        return "remaining Time: NULL"

    time_taken = current_time - starting_time
    steps_taken = i / n_total_steps
    remaining_time = time_taken / steps_taken * (1 - steps_taken)
    return f"remaining Time: {remaining_time:.1f}s"


def progressBarWithTime(i, n_total_steps, starting_time):
    """overwrites the previous progressbar to create a dynamic progressbar with time approximation

    :param i: current step as int
    :param n_total_steps: total steps/batches of the epoch as int
    :param starting_time: starting time of the progressbar
    """
    bar_len = 60
    filled_len = int(round(bar_len * i / float(n_total_steps)))

    status = calculateRemainingTime(starting_time, time.time(), i, n_total_steps)

    percents = round(100.0 * i / float(n_total_steps), 1)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def drawGraph(writer, running_loss, running_correct, n_total_steps, n_total_steps_quarter, epoch, i, predicted):
    """draws graph of accuracy and training loss in tensorboard

    :param writer: writer for the Tensorboard
    :param running_loss: sum of all loses from the learning
    :param running_correct: sum of all correct guesses
    :param n_total_steps: total steps/batches of the epoch as int
    :param n_total_steps_quarter: variable to determine after every x steps add scalar to writer
    :param epoch: current epoch as int
    :param i: current step as int
    :param predicted: the predicted class after input into the model
    """
    writer.add_scalar('training loss', running_loss / n_total_steps_quarter, epoch * n_total_steps + i)
    running_accuracy = running_correct / n_total_steps_quarter / predicted.size(0)
    writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
    writer.close()


def calculateAcc(n_correct, n_wrong, class_name):
    """calculates the accuracy of guessing the class right from the given arrays

    :param n_correct: number of correct guesses as int
    :param n_wrong: number of wrong guesses as int
    :param class_name: name of the individual guessed class
    """
    acc = 100.0 * n_correct / (n_wrong + n_correct)
    print(f'Accuracy of the network on {class_name}: {acc} % ({n_correct}/{n_wrong + n_correct})\n')


def countCorrectlyGuessed(labels, predicted, n_correct_array, n_wrong_array, output_examples):
    """compares the two tensors labels and predicted.
    Counts how many elements of the two given tensors are the same

    :param labels: tensor with class_id as its elements
    :param predicted: tensor with class_id as its elements (return value of model)
    :param n_correct_array: List[int] acts as counter for every right guess for each class
    :param n_wrong_array: List[int] acts as counter for every wrong guess for each class
    :param output_examples: List[String] containing all wrong examples
    :return: returns updated n_correct_array and n_wrong_array
    """
    for batchNumber in range(predicted.size(dim=0)):
        if labels[batchNumber] == predicted[batchNumber]:
            n_correct_array[predicted[batchNumber]] += 1
        else:
            n_wrong_array[labels[batchNumber]] += 1
            output_examples.append(f"predicted: {predicted}\nlabels: {labels}\nWrong\n")

    return n_correct_array, n_wrong_array, output_examples


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
    :param FILE: path of the model to be loaded as a String
    :return: returns the fully loaded model
    """
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load(FILE))
    model.eval()
    return model
