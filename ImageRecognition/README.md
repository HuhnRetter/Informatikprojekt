# Image Recognition

The image recognition uses the given images in the TestImages folder to classify each of them as one of the following classes: "dogs", "flowers", "other"

## Dataset

The training dataset consists of **31.200** images for all classes and **10.400** images for each class. The test dataset consists of **6.000** images for all classes and **2.000** images for each class. After inputting them into the data loader the images get resized to **224x224**.

Here are some examples for each class:

![example](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\Images\example.png)

## Convolutional Neural Network

I choose a pretrained model, because I have already created a  convolutional neural network for the text recognition. Another reason is that I would need to create over 50 different convolutional layers to achieve a good feature extraction. So I choose to use the pretrained **resnet18** model and keeping the weights for feature extraction to minimize the training session. 

I used to the following code snippet to keep the weights from the pretrained model and add a linear layer for the classification:

```python
model = models.resnet18(pretrained=True)
# so not the whole neural net gets rebalanced
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
return model
```

### Parameters

```python
# Parameter
num_classes = len(all_classes)
num_epochs = 3
batch_size = 3
learning_rate = 0.01
```

From multiple tests with different `batch_size` values, and I concluded that the value of `batch_size` needs to be equal or smaller than the classes to minimize overfitting. I chose for the image recognition a higher `learning_rate` than the other projects, because of the size of the dataset and to reduce the computation time. For the same reason I only trained the model for three epochs.

### Training results

The following confusion matrix shows the accuracy for each class:

![ConfusionmatrixTrain](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\Images\ConfusionmatrixTrain.png)

The average image recognition accuracy is 97,67%. The trained model does not show any difficulties at predicting the classes.

### Testing results

The following confusion matrix shows the accuracy for each class:

![ConfusionmatrixTest](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\Images\ConfusionmatrixTest.png)

The average image recognition accuracy is 94,33%. The trained model shows difficulties in predicting "other". The reason for this could be that the model wasn't trained for enough epochs. In almost all cases the model is generalized enough to differentiate between flowers, dogs and others.

## Image Recognition

The following PNG shows an example of how the image recognition looks like to the user:

![ImageRecognition](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\Images\ImageRecognition.png)

The user need to adjust the following parameters in case he wants to use a different trained model or a different folder:

```python
# Parameters
TESTFOLDER = './TestImages'
MODELFOLDER = './Models/'
MODELFILE = "ImageTransferLearningNE3BS3LR001ACC94.pth"
FILENEURALNET = MODELFOLDER + MODELFILE
all_classes = ["dog", "flower", "other"]

# batch_size must be same as images
batch_size = 9
```

The `batch_size` needs to be the same size as images for testing, otherwise there will be a faulty output of the plot.

After adjusting the parameter, the user need to be put his test images inside the TestImages folder. Inside of the TestImages folder they need to be put inside any of the folders, so they can be read with ImageFolder.

```python
test_dataset = ImageFolder(root=TESTFOLDER, transform=multitransform)
```

With the usage of ImageFolder, multiple images can be used for the image recognition. In the next step the data loader iterates through all test images and inserts them into the model. After that a plot containing the test images with guessed labels will be displayed.