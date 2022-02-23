# Text Recognition

The letter recognition uses a convolutional neural net with 2 convolutional layer and pooling layer for each convolutional layer. The input size for the conv net is a grayscale image of size 28x28. For the text recognition multiple letters are combined to words and then the similarity of the guessed word and the actual words are compared.

## Dataset

For training and testing I used the EMNIST Dataset for letters, which contains images of handwritten letters. Here are some examples:



## Convolutional Neural Net

I chose to create an own convolutional neural net, because EMNIST Dataset for letters are grayscale images and only 28x28. Another reason is that most of the models that are used for transfer learning have RGB images with size 224x224 as an input and I would need to unnecessary modify the EMNIST Dataset images.

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        # -> n, 1, 28, 28
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 12, 12
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 4, 4
        x = x.view(-1, 16 * 4 * 4)           # -> n, 256
        x = F.relu(self.fc1(x))              # -> n, 120
        x = F.relu(self.fc2(x))              # -> n, 84
        x = self.fc3(x)                      # -> n, 26
        return x
```

I used the following formula for determining the size of each convolutional layer
$$
(W-F + 2P)/S+1 = Convolutional Layer Size
$$

$$
W = input size; F = filter size; P = padding size; S = stride size
$$

example for the first feature extraction: 
$$
input image-> convolutional         layer -> relu function -> pooling layer
$$

$$
-> 1, 28, 28 ->outputchannelsize, (28-5)/1 + 1 = 24, 24 -> 6, 24, 24 -> 6, 24/2 =         6,12, 12
$$

after the feature extraction the graph is flattened and 3 linear layers are being used for the class classification      

![ConfusionmatrixTrain](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\TextRecognition\Images\ConfusionmatrixTrain.png)

![ConfusionmatrixTest](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\TextRecognition\Images\ConfusionmatrixTest.png)

The confusion matrixes of both training and testing dataset are very similar to each other. Both have difficulty to determine the difference between L and I. The reason for this could be that in some fonts L and I look very similar, for example: I and l (big I and small L).

## Text Recognition

For the text recognition the user needs to input an image containing letters/words. In the newest update the text recognition works for multiple words in the same row. To determine the distance between each letter in a word the user needs to configurate the following parameter:

```python
word_distance = 3 # the higher the gaps between each letter -> the higher the word_distance
```

In the beginning of the text recognition the script needs to find every letter. For this process it determines bounding boxes for each letter and saves the data for later usage. Now each bounding box (letter) will be cropped out of the original image and gets a black border, which size correlates to the following parameter:

```python
padding = 0.3 # percentage from given image
```

I found out that 0.3 is the perfect value for the black border around the letter. After modifying each letter and inputting it into the model, the bounding box of the letter with the label is displayed for the user. In the next process all bounding boxes are being grouped into words by sorting the letters by x and y coordinates and comparing them to determine the distance between each letter. After the letters are combined into a word each letter is compared to `all_possible_words` . If the size is the same and more than `word_min_accuracy` percent are the same letters the script recognizes the word. It is possible to add more words to the `all_possible_words` list.