import cv2
import numpy as np
import skimage.color
import torch
from skimage import color
from skimage import draw
from skimage import io
from skimage import measure
from skimage.transform import resize

import LetterNeuralNet

# Parameters
all_classes = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z"]

FILETEXTTEST = "./testimages/Blacktext/Test.png"
FILENEURALNET = "LetterNeuralNetNE5BS26LR0001LetterNeuralNetACC92.pth"
# following parameters need to be customized to the letter size
area_size_param = 500
padding = 10


def setup():
    """reads an image from a given Path

    :return: returns img as ndarray
    """
    img = io.imread(FILETEXTTEST)
    print(type(img))
    return img


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


def drawBondingBox(img):
    """draws a bonding Box for each letter

    :param img: image as ndarray
    :return: returns the img with drawn bonding boxes
    """
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
            try:
                draw.set_color(img, draw.rectangle_perimeter((x, y), (xmax, ymax)), (0, 100, 0, 1))
            except Exception as e:
                draw.set_color(img, draw.rectangle_perimeter((x, y), (xmax, ymax)), (0, 100, 0))
            cv2.putText(img, letter, (y, x - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 0), 1)
    return img


def setupModel():
    """loads the model from the given path

    :return: returns the loaded model of type LetterNeuralNet.ConvNet
    """
    model = LetterNeuralNet.ConvNet()
    model.load_state_dict(torch.load(FILENEURALNET))
    model.eval()
    return model


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
    img = drawBondingBox(img)
    showimage(img)
