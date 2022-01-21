import sys

import PIL.ImageOps
import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import skimage.color
from skimage.transform import resize
from skimage import measure
from skimage import filters
from skimage import draw

import LetterNeuralNet
import torchvision.transforms as transforms

# Parameters
FILETEXTTEST = "Test.png"
# FILENEURALNET = "LetterNeuralNetNE100BS100LR0001LetterNeuralNet.pth"
# FILENEURALNET = "LetterNeuralNetNE35BS100LR0001LetterNeuralNetACC929375.pth"
FILENEURALNET = "LetterNeuralNetNE5BS26LR0001LetterNeuralNetACC92.pth"
# FILENEURALNET = "LetterNeuralNetNE5BS13LR0001LetterNeuralNetACC925.pth"
# FILENEURALNET = "LetterNeuralNetNE3BS6LR0001LetterNeuralNetACC9169.pth"

FILENEURALNET2 = "LetterNeuralNetNE35BS100LR001LetterNeuralNetACC9115.pth"
# FILENEURALNET = "LetterNeuralNetNE35BS50LR001LetterNeuralNetACC89.pth"
all_classes = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z"]
padding = 20
model_number = 1
area_size_param = 500

# all colors above certain values turn to black
fig, axes = plt.subplots(nrows=5, ncols=5)

ax = axes.ravel()


def setup():
    # img = cv2.imread(FILETEXTTEST)
    img = io.imread(FILETEXTTEST)
    # object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    object_detector = cv2.createBackgroundSubtractorMOG2()
    return img, object_detector


def showimage(img):
    while (1):
        cv2.imshow('text', img)
        # Text Detection if there is background
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


# def showimage(img, title="default"):
#     ax[0].imshow(img)
#     ax[0].set_title(title)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         return

def getContourArea(cnt):
    x, w, y, h = getBoundingBox(cnt)
    return (w - x) * (h - y)


def getBoundingBox(cnt):
    x = np.min(cnt[:, 0])
    xmax = np.max(cnt[:, 0])
    y = np.min(cnt[:, 1])
    ymax = np.max(cnt[:, 1])
    print(f"x:{x} ; y:{y} ; xmax:{xmax}, ymax:{ymax}")
    return x, xmax, y, ymax


def drawBondingBox(img, object_detector):
    model = setupModel()
    outputString = ""
    # remove alpha channel
    try:
        gray_img = color.rgba2rgb(img)
        gray_img = color.rgb2gray(gray_img)
    except Exception as e:
        gray_img = color.rgb2gray(img)
    # showimage(gray_img)
    # mask = object_detector.apply(gray_img)
    # showimage(mask)
    # _, mask = cv2.threshold(mask, 191, 255, cv2.THRESH_BINARY)
    # showimage(mask)

    # blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    blur = filters.gaussian(gray_img)
    showimage(blur)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # showimage(th3)
    # invert
    # th3 = cv2.bitwise_not(th3)
    th3 = skimage.util.invert(blur)
    showimage(th3)
    # contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = measure.find_contours(th3)
    # print(contours)

    i = 0
    for cnt in contours:
        # area = cv2.contourArea(cnt)
        area = getContourArea(cnt)
        # To remove inside boundig rect
        # if hierarchy[0, i, 3] == -1 and area > 10:
        print(f"boundingBoxArea:{area}")
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
                draw.set_color(img, draw.rectangle_perimeter((x, y), (xmax, ymax)), (0, 255, 0, 1))
            except Exception as e:
                draw.set_color(img, draw.rectangle_perimeter((x, y), (xmax, ymax)), (0, 255, 0))
            cv2.putText(img, letter, (y, x - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
            # showimage(img)
        # Show image
        i += 1
    return img, object_detector, outputString


def setupModel():
    if model_number == 1:
        model = LetterNeuralNet.ConvNet()
        model.load_state_dict(torch.load(FILENEURALNET))
        model.eval()
        return model
    elif model_number == 2:
        model = LetterNeuralNet.ConvNet2()
        model.load_state_dict(torch.load(FILENEURALNET2))
        model.eval()
        return model


def getCrop(img, x, y, xmax, ymax):
    return img[x:xmax, y:ymax]


def useLetterRecognition(crop_img, model):
    # showimage(crop_img)
    crop_img = resize(crop_img, (28, 28), anti_aliasing=True)
    # showimage(crop_img)


    numpy_crop_img = np.array(crop_img)
    numpy_crop_img = np.around(numpy_crop_img, 4)
    # print(numpy_crop_img)
    crop_img_tensor = torch.tensor(numpy_crop_img, dtype=torch.float32)
    crop_img_tensor = crop_img_tensor.unsqueeze(dim=0)
    # print(crop_img_tensor)
    #print(crop_img_tensor.shape)
    c_crop_img_tensor = crop_img_tensor.unsqueeze(0)
    # print(c_crop_img_tensor.shape)
    # print(c_crop_img_tensor.type())
    # print(c_crop_img_tensor.dtype)
    print(c_crop_img_tensor)

    with torch.no_grad():
        outputs = model(c_crop_img_tensor)
    print(outputs)
    predicted = torch.argmax(outputs)
    print(all_classes[predicted])
    print("\n")
    return all_classes[predicted]


def getLetter(img, x, y, xmax, ymax, model):
    crop_img = getCrop(img, x, y, xmax, ymax)
    # showimage(crop_img)
    return useLetterRecognition(crop_img, model)


if __name__ == "__main__":
    img, object_detector = setup()
    img, object_detector, outputString = drawBondingBox(img, object_detector)
    showimage(img)
