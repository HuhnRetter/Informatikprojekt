from HelperFunctions.HelperFunctions import *


def trainingPhase(model, criterion, optimizer, train_loader, num_epochs, print_every_x_percent, save, device,
                  confusionmatrixdevice, writer, FILE, all_classes, label_start_at):
    """trains the given model with the given parameters.

    iterates once through the train_loader in each epoch
    and updates the weights in the model

    After every x step update the acc and loss graph in Tensorboard
    and after every epoch create Confusion matrix

    :param model: current model of the class NeuralNet
    :param criterion: loss function from torch.nn.modules.loss (for Example CrossEntropyLoss)
    :param optimizer: optimizer from torch.optim (for Example Adam)
    :param train_loader: dataloader with Training dataset
    :param num_epochs: number of training epochs
    :param print_every_x_percent:
    :param save: parameter to determine if the model should be saved --> 1 == true & 0 == false
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param confusionmatrixdevice: device on which the tensors are being processed for creating the confusionmatrix example: "cuda", "cpu", "dml"
    :param writer: writer for the Tensorboard
    :param FILE: path of the model to be loaded as a String
    :param all_classes: list of all possible classes
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :return: returns trained model of the class NeuralNet
    """
    n_total_steps = len(train_loader)
    n_every_steps = n_total_steps * print_every_x_percent
    running_loss = 0.0
    running_correct = 0
    starting_time = time.time()
    model.to(device)
    for epoch in range(num_epochs):
        if epoch + 1 > 1:
            print("\n####################################Training Resumed#######################################\n")

        for i, (hsl, labels) in enumerate(train_loader):
            if label_start_at == 1:
                # Transform labels if start with 1
                labels = torch.add(labels, -1)
            # Forward pass
            outputs = model(hsl.to(device))
            labels = convertFloatTensorToLongTensor(labels).to(device)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            if (i + 1) % n_every_steps == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Accuracy: {running_correct / n_every_steps / predicted.size(0):.4f}, Loss: {loss.item():.4f}                                                  \n\r')
                ############## TENSORBOARD ########################
                drawGraph(writer, running_loss, running_correct, n_total_steps, n_every_steps, epoch, i, predicted)
                running_correct = 0
                running_loss = 0.0
                ###################################################
            progressBarWithTime(i + n_total_steps * epoch, n_total_steps * num_epochs, starting_time)
        # Save confusion matrix to Tensorboard
        if epoch + 1 == num_epochs:
            print("\n\n####################################Training Completed#####################################")
        else:
            print("\n\n####################################Training Paused########################################")
        currentConfusionMatrix = createConfusionMatrix(train_loader, model, all_classes, label_start_at, 0,
                                                       confusionmatrixdevice)
        plt.show()
        writer.add_figure(f"Confusion matrix training from: {FILE}", currentConfusionMatrix, epoch)
        writer.close()

    model.to(torch.device("cpu"))
    if save == 1:
        torch.save(model.state_dict(), FILE)
    return model


def testingPhase(model, test_loader, writer, FILE, all_classes, label_start_at, device, output_examples_check):
    """tests the model

    outputs Acc of all classes and creates a Confusionmatrix

    :param model: current model of the class NeuralNet
    :param test_loader: dataloader with Test dataset
    :param writer: writer for the Tensorboard
    :param FILE: path of the model to be loaded as a String
    :param all_classes: list of all possible classes
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param output_examples_check: value to enable the output of wrong examples 1 --> True; 0 --> False
    """
    with torch.no_grad():
        print("\n\nStarting with Testing!")

        n_correct_array = list()
        n_wrong_array = list()

        for i in range(len(all_classes)):
            n_correct_array.append(0)
            n_wrong_array.append(0)

        # Console output of the Acc of every class but not as detailed as the confusion matrix

        # Save confusion matrix to Tensorboard
        currentConfusionMatrix = createConfusionMatrix(test_loader, model, all_classes, label_start_at, output_examples_check, device)
        plt.show()
        writer.add_figure(f"Confusion matrix testing from: {FILE}", currentConfusionMatrix)
        writer.close()
