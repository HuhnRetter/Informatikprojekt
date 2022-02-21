import sys

import cv2
import numpy as np
import skimage.color
import torch
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

all_possible_words =["Start","Ziel"]

FILETEXTTEST = "./TestImages/Test7.png"
MODELFOLDER = "./Models/"
MODELPATH = "LetterNeuralNetNE5BS26LR0001ACC92.pth"
FILENEURALNET = f"{MODELFOLDER}{MODELPATH}"
# following parameters need to be customized to the letter size
area_size_param = 500
padding = 10
word_min_accuracy = 0.5


@dataclass
class BoundingBox:
    """For keeping all the information to the corresponding bounding box."""
    letter: str
    x: int
    y: int
    xmax: int
    ymax: int

    # def __init__(self, letter: str, x: int, y: int, xmax: int, ymax: int ):
    #     self.letter = letter
    #     self.x = x
    #     self.y = y
    #     self.xmax = xmax
    #     self.ymax = ymax

    def getYsize(self):
        return self.ymax-self.y

    def getXsize(self):
        return self.xmax-self.x

def setup():
    """reads an image from a given Path

    :return: returns img as ndarray
    """
    img = io.imread(FILETEXTTEST)
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

    # showimage(gray_img)
    blur = gray_img
    # showimage(blur)
    # invert
    th3 = skimage.util.invert(blur)
    # showimage(th3)
    return th3


def setupModel():
    """loads the model from the given path

    :return: returns the loaded model of type LetterNeuralNet.ConvNet
    """
    model = LetterNeuralNet.ConvNet()
    model.load_state_dict(torch.load(FILENEURALNET))
    model.eval()
    return model


def showimage(img):
    """shows given image until ESC is pressed

    :param img: image as ndarray
    """
    while (1):
        cv2.imshow('text', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


def getContourArea(cnt):
    """gets the Area of the Bounding box

    :param cnt: is the contours of a possible letter of type ndarray
    :return: returns area of the Bounding box
    """
    x, w, y, h = getBoundingBox(cnt)
    return (w - x) * (h - y)


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
    for cnt in contours:
        area = getContourArea(cnt)
        print(f"boundingBoxArea for customizing in auto Bounding box:{area}")
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
    crop_img = resize(crop_img, (28, 28), anti_aliasing=True)

    numpy_crop_img = np.array(crop_img)
    flipped_img = np.fliplr(numpy_crop_img)
    turned_img = np.rot90(flipped_img)

    crop_img_tensor = torch.from_numpy(turned_img.copy())
    crop_img_tensor = crop_img_tensor.unsqueeze(dim=0)
    c_crop_img_tensor = crop_img_tensor.unsqueeze(0)
    c_crop_img_tensor = c_crop_img_tensor.float()

    with torch.no_grad():
        outputs = model(c_crop_img_tensor)
    predicted = torch.argmax(outputs)
    return all_classes[predicted]

def differenceForBoundingbox(prev, current, axis, diff_param):
    if axis == "y":
        return abs(prev.ymax - current.y) > (prev.getYsize() * diff_param)
    else:
        return abs(prev.xmax - current.xmax) > (prev.getXsize() * diff_param)


def sortListOfBoundingboxBy(diff_param, key_sort_param, all_boundingboxes, axis):
    #print(f"before all_boundingboxes: {all_boundingboxes}")

    all_boundingboxes.sort(key=key_sort_param)
    #print(f"after all_boundingboxes: {all_boundingboxes}")

    wordlist = list()
    wordlist.append(all_boundingboxes[0])
    # separate words with , on given axis
    for b in range(1, len(all_boundingboxes)):
        if differenceForBoundingbox(all_boundingboxes[b-1], all_boundingboxes[b], axis, diff_param) and all_boundingboxes[b].letter != ",":
            wordlist.append(BoundingBox(",", -1, -1, -1, -1))
            wordlist.append(all_boundingboxes[b])
        else:
            wordlist.append(all_boundingboxes[b])
    #print(f"wordlist: {wordlist}")
    return wordlist

def splitList(all_boundingboxes):
    # Create an empty list
    list_of_lists = []

    #splittedList = list(list)
    counter = 0
    list_of_lists.append([])
    for boundingbox in all_boundingboxes:
        if(boundingbox.letter == ","):
            counter+=1
            list_of_lists.append([])
        else:
            list_of_lists[counter].append(boundingbox)

    return list_of_lists

def transformBoundingboxList(sorted_boundingboxes_by_words):

    wordBoundingboxlist = list()

    for word in sorted_boundingboxes_by_words:
        wordString = ""
        x = sys.maxsize
        y = sys.maxsize
        xmax = 0
        ymax = 0
        for boundingbox in word[0]:
            wordString = wordString + boundingbox.letter
            if(boundingbox.x < x):
                x = boundingbox.x
            if (boundingbox.y < y):
                y = boundingbox.y
            if (boundingbox.xmax > xmax):
                xmax = boundingbox.xmax
            if (boundingbox.ymax > ymax):
                ymax = boundingbox.ymax

        wordBoundingboxlist.append(BoundingBox(wordString,x,y,xmax,ymax))

    return wordBoundingboxlist

def guessWord(word: str):

    accuracy_to_words = list()

    for i,w in enumerate(all_possible_words):
        accuracy_to_words.append(0)
        #not same length = cant be the word
        if(len(word) != len(w)):
            continue

        same_letter_counter = 0
        for letterindex,notUsed in enumerate(word):
            if(word[letterindex] == w[letterindex].upper()):
                same_letter_counter+=1
        accuracy_to_words [i] = same_letter_counter/len(word)

    #choose a word from the given accuracy and return the highest possibility
    highest_index = 0
    for i,notUsed in enumerate(accuracy_to_words):
        if accuracy_to_words[i] > accuracy_to_words[highest_index]:
            highest_index = i

    if(accuracy_to_words[highest_index] >= word_min_accuracy):
        return all_possible_words[highest_index]

    else:
        return "other"


def getWords(all_boundingboxes):
    #sorts array for words and returns the sorted array with words

    wordlistX = sortListOfBoundingboxBy(0.1, operator.attrgetter('x'), all_boundingboxes, "x")

    #split list by , for input
    splittedList = splitList(wordlistX)


    # Create an empty list
    wordlistY = []

    counter = 0
    for word in splittedList:
        returnwordlist = (sortListOfBoundingboxBy(1, operator.attrgetter('ymax'), word, "y"))

        splittedReturnwordlist = splitList(returnwordlist)
        #print(f"splittedReturnwordlist: {splittedReturnwordlist}")
        #to remove list stacking
        for boundingbox in splittedReturnwordlist:
            wordlistY.append([])
            wordlistY[counter].append(boundingbox)
            #print(f"wordlistY: {wordlistY}")
            counter+=1

    #transform each boundboxlist (word)
    transformedlist = transformBoundingboxList(wordlistY)

    # return the sorted words as an a boundbox list
    return transformedlist




def drawBoundingBoxForWord(img, all_boundingboxes):
    #draws bounding box around the words
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
    showimage(b_img)
    w_img = drawBoundingBoxForWord(copy_img, all_boundingboxes)
    showimage(w_img)
