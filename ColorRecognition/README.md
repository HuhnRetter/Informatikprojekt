# Color Recognition

The color recognition is based on a simple neural network with one hidden layer. The model is trained with the HSL-Values of a color instead of the BGR-Values to have a better comparison between manual (predetermined by human) and automatic (weights learned by neural network) color detection. For the color detection only the H and L Values from HSL are being used, because they are sufficient for simple color recognition. The trained colors consists of white, red, green and "other". The following code snippet shows an example of the predetermined weights for the color red and white.

```python
if light_value > 229:
    return "WHITE"
elif hue_value < 5:
    return "RED"
```

## Dataset

The dataset was created by randomly generating a color with BGR-Values and manually determining which color class they belong to. After that each BGR-Value was transformed into a HSL-Value. The training dataset consists of **1600** color samples with **400** colors per class. The testing dataset consists of **400** color samples with **100** colors per class. 

Here are examples for each color from the TrainingDatasetHLS:

| class_id | H-Value             | L-Value             |
| -------- | ------------------- | ------------------- |
| 0        | 0.7816091954022988  | 0.9235294117647059  |
| 1        | 0.9166666666666666  | 0.45098039215686275 |
| 2        | 0.40454545454545454 | 0.7294117647058823  |
| 3        | 0.5851851851851851  | 0.7470588235294118  |

## Neural net

I chose a simple neural network because of the low complexity in color recognition. The following code snippet out of the `NeuralNet` class shows that there is only one hidden layer.

```python
self.l1 = nn.Linear(input_size, hidden_size)
self.relu = nn.ReLU()
self.l2 = nn.Linear(hidden_size, num_classes)
```

### Parameters

I used the following parameters for my color recognition model:

```python
# Parameter
input_size = 2
hidden_size = 25
num_classes = len(all_classes) # 4
num_epochs = 150
batch_size = 4
learning_rate = 0.001
```

I tested the model with different parameters and in conclusion I choose the parameters for the following reason. The `hidden_size = 25` is the lowest possible value, which results in the highest possible accuracy from the given dataset. The result of multiple tests with different `batch_size` values resulted in the following conclusion, that the value of `batch_size` needs to be equal or smaller than the classes to minimize overfitting. 

### Training results

The following confusion matrix shows the accuracy for each color:

![ConfusionmatrixTrain](https://github.com/HuhnRetter/Informatikprojekt/blob/main/ColorRecognition/Images/ConfusionmatrixTrain.png)

The average color recognition accuracy is 94,25%. The trained model shows difficulties in predicting the color class "other". There are multiple possible reasons for this low value in comparison to the other color classes. One of them is that the dataset is too small or the variety of "other" colors is too low. Another reason could be that not using the S-Value of HSL results in a lower accuracy.

### Testing results

The following confusion matrix shows the accuracy for each color:

![ConfusionmatrixTest](https://github.com/HuhnRetter/Informatikprojekt/blob/main/ColorRecognition/Images/ConfusionmatrixTest.png)

The average color recognition accuracy is 87,75%. The trained model shows a good generalization for the color recognition of red, green and "other". The color recognition of the white color seems to have a rather bad generalization in comparison to the other colors. There are multiple possible reasons for this problem. One of them could be that the balancing of the white colors for the training dataset is bad. This could mean that the dataset does not cover all possible shades of white, which results in a worse accuracy of white with the testing dataset. Another reason could be that because I manually selected each color, there is a some noise in the testing or training dataset. Both possible reasons would support why the white colors are in 20% of the cases guessed as "other". 

## Color Recognition

The following GIF shows an example of how the color recognition looks like to the user:

![ColorRecognition](https://github.com/HuhnRetter/Informatikprojekt/blob/main/ColorRecognition/Images/ColorRecognition.gif)

### Controls for the user

The user can choose any image out of the test folder and needs to modify the `TESTPATH` variable to match the path of the image. Also the user can choose between the different trained models by modifying the `MODELPATH` variable to match the path of the model. If the chosen model has a different `hidden_size` that given, the user also needs to modify it to properly load the model.

After choosing the image and model a window opens the chosen image. At the start in the left and right corner are fully black rectangles. After the user chooses a pixel with his mouse and clicks the left mouse button, a color name appears in both rectangles. The left rectangle shows the predicted color from the trained model and the right rectangle shows the predicted color from the predetermined weights (manually chosen).

The user needs to press ESC for exiting the color recognition window.

### Parameters

```python
# Model Settings
############################
TESTFOLDER = "./TestImages/"
TESTPATH = "color_test2.jpg"
FILECOLORTEST = f"{TESTFOLDER}{TESTPATH}"

MODELFOLDER = "./Models/"
# MODELPATH = "ColorNeuralNetHS25NE10BS10HLSACC9075.pth"
# MODELPATH = "ColorNeuralNetHS25NE1500HLSACC8775.pth"
MODELPATH = "ColorNeuralNetHS25NE150BS4LR0001HLS.pth"
FILE = f"{MODELFOLDER}{MODELPATH}"
input_size = 2
hidden_size = 25
num_classes = 4
############################
```