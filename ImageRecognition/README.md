# Image Recognition

The image recognition is using the pretrained resnet18 model and keeping the weights for feature extraction. The model was trained on the following classes: "dogs", "flowers", "other". 

## Dataset

The training dataset consists of 31.200 images for all classes and 10.400 images for each class. The test dataset consists of 6000 images for all classes and 2000 images for each class. Both datasets (training and testing) consists of multiple smaller datasets for each class. Here are some examples for each class:

![2008](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\DatasetExample\Dog\2008.jpg)

![7](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\DatasetExample\Flower\7.jpg)

![COCO_train2014_000000000009](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\DatasetExample\Other\COCO_train2014_000000000009.jpg)

## Neural net

I choose a pretrained model, because I have already a convolutional net in the text recognition and I wanted to try transfer learning. For the parameters I chose 3 for the batch size, because of the number of classes. Also I choose a higher learning rate than the other Projects to minimize the training session. With my current settings the model was trained over 5 hours.

![ConfusionmatrixTrain](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\Images\ConfusionmatrixTrain.png)

![ConfusionmatrixTest](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ImageRecognition\Images\ConfusionmatrixTest.png)

As we can see the confusion matrix of training and test datasets are pretty similar to each other. The accuracy for the class "other" is in the test datasets slightly less than in the training dataset (from 96% to 86%). All in all it is still a solid model for image recognition 

## Image Recognition

The images to be used in the image recognition, they need to be put inside the TestImages folder. Inside of the TestImages folder they need to be put inside any of the folders, so they can be read with ImageFolder.

```python
test_dataset = ImageFolder(root=TESTFOLDER, transform=multitransform)
```

With the usage of ImageFolder, you can multiple images for testing. It's really important that the batch_size is equal to the number of tested images for the mathplot. Otherwise the script could fail. After inputting all test images into the model, a mathplot containing the test images with guessed labels will be displayed.