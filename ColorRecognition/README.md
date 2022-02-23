# Color Recognition

The color recognition is based on a simple neural network with one hidden layer. The neural net is trained by the HSL-Values of a color instead of the BGR-Values to have a better comparison between manual (predetermined by human) and automatic (weights learned by neural net) color detection. For the Color detection only the H and L Values from HSL are being used because they are sufficient for simple color recognition. 

## Dataset

The dataset was created by randomly generating a color and me manually determining which color class they belong, because of that there are only 400 datasets for each class.  ![ConfusionmatrixTest](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ColorRecognition\Images\ConfusionmatrixTest.png)

It could be possibly because I choose each color manually there is some noise in the datasets. Also we can see that possibly because of that white has only an accuracy of 75% and 20% being wrongly guessed as a "other" color. But still the most important colors red, green, other are mostly being guessed right. In conclusion white is being guessed right if it is clear white, but it is light grey or any other variant of white they model has its problems.

## Neural net

I choose a simple neural network because there isn't much complexity in color recognition and because of that I tried different parameter for this simple NN. I found out that they batch size needs to be equal or smaller than the classes to minimize overfitting and because of that I choose 4 as my batch size for most of my models. Also to minimize overfitting I chose the hidden size to be as small as possible with 25. 

![](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ColorRecognition\Images\ConfusionmatrixTest.png)![ConfusionmatrixTrain](U:\Studium\5.Semester\Informatikprojekt\Informatikprojekt\ColorRecognition\Images\ConfusionmatrixTrain.png)

If we compare the test and training confusion matrix, there is a clear difference between the white accuracy in both of them. The cause of this could be noise in the white dataset.

## Color Recognition

The color recognition python script uses a given image and gives the user the possibility to choose a pixel for the color recognition. In the top corners of the window they are both ways of color recognition to demonstrate the difference between the model and the weights chosen by a human.