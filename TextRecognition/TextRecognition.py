import sys
import cv2
import numpy as np
import skimage.color
import torch
from matplotlib import pyplot as plt
from skimage import color
from skimage import draw
from skimage import io
from skimage import measure
from skimage.transform import resize
from dataclasses import dataclass
import operator
import LetterNeuralNet

# Parameters
all_classes = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z"]

all_possible_words = ["Start", "Ziel"]

FILETEXTTEST = "./TestImages/Test8.png"
MODELFOLDER = "./Models/"
MODELPATH = "LetterNeuralNetNE5BS26LR0001ACC92.pth"
FILENEURALNET = f"{MODELFOLDER}{MODELPATH}"
# following parameters need to be customized to the letter size
padding = 0.3  # percentage from given image
word_min_accuracy = 0.5
line_diff = 0.1
word_distance = 10  # gap between each letter as pixels


@dataclass
class BoundingBox:
    """For keeping all the information to the corresponding bounding box."""
    letter: str
    x: int
    y: int
    xmax: int
    ymax: int

    def getYsize(self):
        return self.ymax - self.y

    def getXsize(self):
        return self.xmax - self.x


def setup():
    """reads an image from a given Path

    :return: returns img as ndarray
    """
    img = io.imread(FILETEXTTEST)
    showimage(img, "original image")
    return img


def setupImage(img):
    """does multiple modification to the original img

    converts the img to a grayscale image
    blurs the img to get better bounding box results
    inverts the img, so it can be used by the function measure.find_contours

    :param img: image as ndarray
    :return: returns the modified image as ndarray
    """
    # remove alpha channel
    try:
        gray_img = color.rgba2rgb(img)
        gray_img = color.rgb2gray(gray_img)
    except Exception as e:
        gray_img = color.rgb2gray(img)

    showimage(gray_img, "gray_scale", 'gray')
    # gaussian blur was removed because if the letters are to close to each other they don't get recognized as 2
    # different letters
    blur = gray_img
    # showimage(blur)
    # invert
    th3 = skimage.util.invert(blur)
    showimage(th3, "inverted gray_scale", 'gray')
    return th3


def setupModel():
    """loads the model from the given path

    :return: returns the loaded model of type LetterNeuralNet.ConvNet
    """
    model = LetterNeuralNet.ConvNet()
    model.load_state_dict(torch.load(FILENEURALNET))
    model.eval()
    return model


def showimage(img, title, cmap=None):
    """shows given image as a plot with a title

    :param img: image as ndarray
    :param title: title of the plot
    :param cmap: cmap of the plot
    """
    plt.imshow(img, cmap)
    plt.title(title)
    plt.show()


def getContourArea(cnt):
    """gets the Area of the Bounding box

    :param cnt: is the contours of a possible letter of type ndarray
    :return: returns area of the Bounding box
    """
    x, w, y, h = getBoundingBox(cnt)
    return (w - x) * (h - y)


def getAreaParam(contours):
    """gets the AreaParam for filtering out bounding boxes of the insides of letters (A,b,d,...)

    :param contours: list of all contours --> used to determine AreaParam for eliminating all contours inside letters
    :return: returns a value for eliminating all contours inside letters
    """
    highest = 0
    for cnt in contours:
        area = getContourArea(cnt)
        if (highest < area):
            highest = area
    return highest / 5.5


def getBoundingBox(cnt):
    """gets the bounding box of the given contour

    :param cnt: is the contours of a possible letter of type ndarray
    :return: returns an interval of the bounding box in form of four variables -> x, xmax, y, ymax
    """
    x = np.min(cnt[:, 0])
    xmax = np.max(cnt[:, 0])
    y = np.min(cnt[:, 1])
    ymax = np.max(cnt[:, 1])
    return x, xmax, y, ymax


def getCrop(img, x, y, xmax, ymax):
    """gets a cropped image for the given coordinates

    :param img: image as ndarray
    :param x: min value of x-coordinate of type int
    :param y: min value of y-coordinate of type int
    :param xmax: max value of x-coordinate of type int
    :param ymax: max value of y-coordinate of type int
    :return: returns the cropped image as ndarray
    """
    return img[x:xmax, y:ymax]


def drawBoundingBox(img):
    """draws a bounding Box for each letter

    :param img: image as ndarray
    :return: returns the img with drawn bonding boxes
    """
    all_boundingboxes = list()

    model = setupModel()
    th3 = setupImage(img)
    contours = measure.find_contours(th3)

    area_size_param = getAreaParam(contours)

    for cnt in contours:
        area = getContourArea(cnt)
        # print(f"boundingBoxArea for customizing in auto Bounding box:{area}")
        if area > area_size_param:
            x, xmax, y, ymax = getBoundingBox(cnt)
            x -= padding
            y -= padding
            xmax += padding
            ymax += padding
            x = int(x)
            y = int(y)
            xmax = int(xmax)
            ymax = int(ymax)
            letter = getLetter(th3, x, y, xmax, ymax, model)
            # save parameters of the bounding box with corresponding letter
            all_boundingboxes.append(BoundingBox(letter, x, y, xmax, ymax))

            try:
                draw.set_color(img, draw.rectangle_perimeter((x, y), (xmax, ymax)), (0, 100, 0, 1))
            except Exception as e:
                draw.set_color(img, draw.rectangle_perimeter((x, y), (xmax, ymax)), (0, 100, 0))
            cv2.putText(img, letter, (y, x - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 0), 1)
    return img, all_boundingboxes


def useLetterRecognition(crop_img, model):
    """uses the cropped image to guess a letter with the given model

    :param crop_img: cropped image as ndarray
    :param model: loaded model of type LetterNeuralNet.ConvNet
    :return: returns the guessed letter as a string
    """

    fig, axs = plt.subplots(1, 4)

    # adding a black border around the letter
    axs[0].imshow(crop_img, "gray")
    axs[0].set_title("cropped image", fontsize=10)

    bt = round(28 * (1 - padding))
    color_name = 0
    value = [color_name for i in range(3)]
    crop_img = cv2.copyMakeBorder(crop_img, bt, bt, bt, bt, cv2.BORDER_CONSTANT, value=value)
    axs[1].imshow(crop_img, "gray")
    axs[1].set_title("blackborder added", fontsize=10)

    crop_img = resize(crop_img, (28, 28), anti_aliasing=True)

    axs[2].imshow(crop_img, "gray")
    axs[2].set_title("resized", fontsize=10)
    numpy_crop_img = np.array(crop_img)
    flipped_img = np.fliplr(numpy_crop_img)
    turned_img = np.rot90(flipped_img)
    axs[3].imshow(turned_img, "gray")
    axs[3].set_title("axis correction", fontsize=10)

    crop_img_tensor = torch.from_numpy(turned_img.copy())
    print(f"before tensor transformation: {crop_img_tensor.shape}")
    crop_img_tensor = crop_img_tensor.unsqueeze(dim=0)
    c_crop_img_tensor = crop_img_tensor.unsqueeze(0)
    c_crop_img_tensor = c_crop_img_tensor.float()
    print(f"after tensor transformation: {c_crop_img_tensor.shape}")
    with torch.no_grad():
        outputs = model(c_crop_img_tensor)
    predicted = torch.argmax(outputs)
    fig.suptitle(f'Guessed letter: {all_classes[predicted]}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return all_classes[predicted]


def printForBoundingBoxes(firstmessage, all_boundingboxes):
    """prints out all elements from a given list

    :param firstmessage:
    :param all_boundingboxes:
    """
    print(f"\n{firstmessage} \n")
    for boundingbox in all_boundingboxes:
        print(boundingbox)
    print("\n")


def differenceForBoundingbox(prev, current, axis, diff_param):
    """compares distance between two points

    :param prev: previous letter as a boundingbox data class object
    :param current: current letter as a boundingbox data class object
    :param axis: describes on which axis the two boundingbox data class objects are compared (x or y)
    :param diff_param: parameter for setting the allowed interval (if axis == x --> checking same row, if axis == y --> checking if letters belong to the same word)
    :return: returns a boolean value for checking if the letters are on the same/similar row (axis == x) or if the letters belong to the same word (axis == y)

    """
    if axis == "y":
        return abs(prev.ymax - current.y) > diff_param
    else:
        return abs(prev.xmax - current.xmax) > (prev.getXsize() * diff_param)


def sortListOfBoundingboxBy(diff_param, key_sort_param, all_boundingboxes, axis):
    """sorts the List of BoundingBox by the given parameters

    :param diff_param: parameter for setting the allowed interval (if axis == x --> checking same row, if axis == y --> checking if letters belong to the same word)
    :param key_sort_param: parameter by which the list is being sorted
    :param all_boundingboxes: list with all bounding box data class objects
    :param axis: describes on which axis the two boundingbox data class objects are compared (x or y)
    :return: returns sorted List with bounding box data class objects
    """
    printForBoundingBoxes(f"\nallboundingboxes before being sorted on {axis} axis:", all_boundingboxes)
    all_boundingboxes.sort(key=key_sort_param)
    printForBoundingBoxes(f"allboundingboxes after being sorted on {axis} axis:", all_boundingboxes)
    wordlist = list()
    wordlist.append(all_boundingboxes[0])
    # separate words with , on given axis
    for b in range(1, len(all_boundingboxes)):
        if differenceForBoundingbox(all_boundingboxes[b - 1], all_boundingboxes[b], axis, diff_param) and \
                all_boundingboxes[b].letter != ",":
            wordlist.append(BoundingBox(",", -1, -1, -1, -1))
            wordlist.append(all_boundingboxes[b])
        else:
            wordlist.append(all_boundingboxes[b])
    printForBoundingBoxes(f"words split by , :", wordlist)
    return wordlist


def splitList(all_boundingboxes):
    """splits the list if the letter is "," by creating its own list

    :param all_boundingboxes: list with all bounding box data class objects
    :return: returns an interconnected list with boundingboxes
    """
    # Create an empty list
    list_of_lists = []

    counter = 0
    list_of_lists.append([])
    for boundingbox in all_boundingboxes:
        if boundingbox.letter == ",":
            counter += 1
            list_of_lists.append([])
        else:
            list_of_lists[counter].append(boundingbox)

    return list_of_lists


def transformBoundingboxList(sorted_boundingboxes_by_words):
    """transforms multiple letter Bounding boxes to a word Bounding box

    :param sorted_boundingboxes_by_words: sorted list of bounding boxes by words in the following format: [[BoundingBox(letter='S', ...), BoundingBox(letter='T', ...), ...],[BoundingBox(letter='Z', ...), BoundingBox(letter='I', ...), ...],[BoundingBox(letter='H', ...), ...]]
    return: returns transformed list in the following format: [[BoundingBox(letter='START', ...)],[BoundingBox(letter='ZIEL', ...)],[BoundingBox(letter='HI', ...)]]
    """
    wordBoundingboxlist = list()

    for word in sorted_boundingboxes_by_words:
        wordString = ""
        x = sys.maxsize
        y = sys.maxsize
        xmax = 0
        ymax = 0
        for boundingbox in word[0]:
            wordString = wordString + boundingbox.letter
            if boundingbox.x < x:
                x = boundingbox.x
            if boundingbox.y < y:
                y = boundingbox.y
            if boundingbox.xmax > xmax:
                xmax = boundingbox.xmax
            if boundingbox.ymax > ymax:
                ymax = boundingbox.ymax

        wordBoundingboxlist.append(BoundingBox(wordString, x, y, xmax, ymax))

    return wordBoundingboxlist


def guessWord(word: str):
    """compares given word string with the possible words to guess a word by comparing how many letters are the same

    :param word: string of a word (letters of the word were guessed by the letter recognition)
    :return: returns the guessed word
    """
    accuracy_to_words = list()

    for i, w in enumerate(all_possible_words):
        accuracy_to_words.append(0)
        # not same length = cant be the word
        if len(word) != len(w):
            continue

        same_letter_counter = 0
        for letterindex, notUsed in enumerate(word):
            if word[letterindex] == w[letterindex].upper():
                same_letter_counter += 1
        accuracy_to_words[i] = same_letter_counter / len(word)

    # choose a word from the given accuracy and return the highest possibility
    highest_index = 0
    for i, notUsed in enumerate(accuracy_to_words):
        if accuracy_to_words[i] > accuracy_to_words[highest_index]:
            highest_index = i

    if (accuracy_to_words[highest_index] >= word_min_accuracy):
        return all_possible_words[highest_index]

    else:
        return "other"


def getWords(all_boundingboxes):
    """gets the words out of all the bounding boxes

    :param all_boundingboxes: list with all bounding box data class objects
    return: returns a list with all words in the following format: [[BoundingBox(letter='START', ...)],[BoundingBox(letter='ZIEL', ...)],[BoundingBox(letter='HI', ...)]]
    """
    # sorts array for words and returns the sorted array with words

    wordlistX = sortListOfBoundingboxBy(line_diff, operator.attrgetter('x'), all_boundingboxes, "x")

    # split list by , for input
    splittedList = splitList(wordlistX)

    # Create an empty list
    wordlistY = []

    counter = 0
    for word in splittedList:
        returnwordlist = (
            sortListOfBoundingboxBy(word_distance, operator.attrgetter('ymax'), word, "y"))

        splittedReturnwordlist = splitList(returnwordlist)
        # to remove list stacking
        for boundingbox in splittedReturnwordlist:
            wordlistY.append([])
            wordlistY[counter].append(boundingbox)
            counter += 1

    # transform each boundboxlist (word)
    transformedlist = transformBoundingboxList(wordlistY)

    # return the sorted words as a boundbox list
    return transformedlist


def drawBoundingBoxForWord(img, all_boundingboxes):
    """draws the bounding boxes for each word and labels it

    :param img: image as ndarray
    :param all_boundingboxes: list with all bounding box data class objects
    :return: returns the image with the drawn boundingboxes
    """
    # draws bounding box around the words
    words = getWords(all_boundingboxes)

    for word in words:
        guessed_word = guessWord(word.letter)
        print(word)
        try:
            draw.set_color(img, draw.rectangle_perimeter((word.x, word.y), (word.xmax, word.ymax)), (0, 100, 0, 1))
        except Exception as e:
            draw.set_color(img, draw.rectangle_perimeter((word.x, word.y), (word.xmax, word.ymax)), (0, 100, 0))
        cv2.putText(img, guessed_word, (word.y, word.x - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 0), 1)
    return img


def getLetter(img, x, y, xmax, ymax, model):
    """gets a letter from an image and a given interval of a bounding box

    :param img: image as ndarray
    :param x: min value of x-coordinate of type int
    :param y: min value of y-coordinate of type int
    :param xmax: max value of x-coordinate of type int
    :param ymax: max value of y-coordinate of type int
    :param model: loaded model of type LetterNeuralNet.ConvNet
    :return: returns the guessed letter as a string
    """
    crop_img = getCrop(img, x, y, xmax, ymax)
    return useLetterRecognition(crop_img, model)


if __name__ == "__main__":
    img = setup()
    copy_img = np.copy(img)
    b_img, all_boundingboxes = drawBoundingBox(img)
    showimage(b_img, "final image with letter recognition")
    w_img = drawBoundingBoxForWord(copy_img, all_boundingboxes)
    showimage(w_img, "final image with word recognition")
