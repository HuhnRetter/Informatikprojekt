# Text Recognition

The text recognition is split into two phases. The first phase consists of recognizing all possible letters in the given image. In the second phase the letters are grouped into words and compared to the possible words. Here is an example of the possible words:

```python
all_possible_words = ["Start", "Ziel"]
```

For the first phase a model for letter recognition needs to be trained. The letter recognition uses a convolutional neural network with two convolutional layer and pooling layer for each convolutional layer. The input size for the model is a grayscale image of size **28x28**. 

## Dataset

I used the EMNIST Dataset for training a generalized model for letter recognition with different fonts. The EMNIST Dataset for letters contains **145,600** images of handwritten letters. Each image is an grayscale image of size **28x28**. Every letter in the dataset is white with a black background.

Here are some examples:

![output](https://github.com/HuhnRetter/Informatikprojekt/blob/main/TextRecognition/Images/output.png)



## Convolutional Neural Network

I chose to create an own convolutional neural network, because most of the pretrained models have RGB images with size **224x224** as an input and I would need to do unnecessary modifications to the EMNIST Dataset.

The following code snippet of the `ConvNet` class shows the first layers of the convolutional neural network. These two convolutional layers with each having a pooling layers are used for feature extraction from the given image.

```python
x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 12, 12
x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 4, 4
```

After the feature extraction the graph is flattened and three linear layers are being used for the class classification.

```python
x = x.view(-1, 16 * 4 * 4)  # -> n, 256
x = F.relu(self.fc1(x))  # -> n, 120
x = F.relu(self.fc2(x))  # -> n, 84
x = self.fc3(x)  # -> n, 26
```

### 																							Formula for determining the size of each convolutional layer

<img src="https://render.githubusercontent.com/render/math?math=(W-F + 2P)/S+1 = Convolutional Layer Size">

<img src="https://render.githubusercontent.com/render/math?math=W = input size, F = filter size, P = padding size, S = stride size">

### 																																		example: 

<img src="https://render.githubusercontent.com/render/math?math=input image-> convolutionallayer -----------> relu function -> pooling layer">

<img src="https://render.githubusercontent.com/render/math?math=-> 1, 28, 28 ->outputchannelsize, (28-5)/1 + 1 = 24, 24 -> 6, 24, 24 -> 6, 24/2 =         6,12, 12">

### Parameters

I used the following parameters for my letter recognition model:

```python
# Parameter
num_epochs = 10
batch_size = 26
learning_rate = 0.001
```

From multiple tests with different `batch_size` values, and I concluded that the value of `batch_size` needs to be equal or smaller than the classes to minimize overfitting.

### Training results

The following confusion matrix shows the accuracy for each letter:

![ConfusionmatrixTrain](https://github.com/HuhnRetter/Informatikprojekt/blob/main/TextRecognition/Images/ConfusionmatrixTrain.png)

The average letter recognition accuracy is 95,11%. the trained model shows difficulties in predicting "G", "I", "L". The reason for this could be that in some fonts L and I look very similar, for example in this font: I and l (big I and small L). The same applies to the letter "G", in some cases it looks very similar to the letter "Q", because the letters in the dataset are handwritten.

### Testing results

The following confusion matrix shows the accuracy for each letter:

![ConfusionmatrixTest](https://github.com/HuhnRetter/Informatikprojekt/blob/main/TextRecognition/Images/ConfusionmatrixTest.png)

The average letter recognition accuracy is 93%. Both confusion matrices have very similar accuracies for each letter, which concludes that the trained model is generalized for letter recognition. The only exceptions are the previous mentioned letters. 

## Text Recognition

The following GIF shows an example of how the text recognition looks like to the user:

<img src="https://github.com/HuhnRetter/Informatikprojekt/blob/main/TextRecognition/Images/WordRecognition.gif" alt="WordRecognition" style="zoom: 150%;" />

For the text recognition the user needs to input an image containing letters/words. The text recognition works for multiple words in the same row. The user needs to configurate the following parameter to determine the distance between each letter in a word :

```python
word_distance = 10  # gap between each letter as pixels
```

In the beginning of the text recognition the script needs to find every letter. For this process it determines bounding boxes for each letter and saves the data for later usage. If the contour of a possible letter fulfils the following condition `contour_area > area_size_param` . The contour gets transformed into a bounding box. After that it will be cropped out of the original image and gets a black border, which size the black border gets, correlates to the parameter `padding`. After that the `cropped_image` is resized to **28x28**. In the last step the Image gets an axis correction to match the orientation of the training data. In the following the complete process of the letter transformation is shown:

![exampleLetter](https://github.com/HuhnRetter/Informatikprojekt/blob/main/TextRecognition/Images/exampleLetter.png)



```python
padding = 0.3 # percentage from given image
```

I found out that 0.3 is the perfect value for the black border around the letter. After modifying each letter and inputting it into the model, the bounding box of the letter with the label is displayed for the user. 

After all bounding boxes from the letters are plotted, they get grouped into words by sorting the letters by `x` and `ymax` coordinates and comparing them to determine the `word_distance` between each letter and the `line_diff`if the letters aren't exactly on the same `y` coordinate . After the letters are combined into a word, each letter is compared to `all_possible_words` . If the size is the same as a word in`all_possible_words`  and the similarity of all letters is more than `word_min_accuracy` percent, then the script recognizes the word. It is possible to add more words to the `all_possible_words` list.