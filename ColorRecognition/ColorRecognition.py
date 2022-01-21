import dataclasses

import cv2
import numpy as np
import colour_datasets
import torch
from torch import nn
import ColorNeuralNet

mouseX = 0
mouseY = 0
FILECOLORTEST = "color_test2.jpg"
############################
# NeuralNetSettings
# FILE = "ColorNeuralNetHS25NE10BS10HLSACC9075.pth"
FILE = "ColorNeuralNetHS25NE1500HLSACC8775.pth"
input_size = 2
hidden_size = 25
num_classes = 4
############################


def get_MousePos(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONUP:
        mouseX, mouseY = x, y


def setup():
    img = cv2.imread(FILECOLORTEST)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_MousePos)
    showImage(img, setupModel())


def setupModel():
    model = ColorNeuralNet.NeuralNet(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(FILE))
    model.eval()
    return model


def autoColorPicker(outputs):
    predicted = torch.argmax(outputs, 0)
    if (predicted == 0):
        return "WHITE"
    elif (predicted == 1):
        return "RED"
    elif (predicted == 2):
        return "GREEN"
    elif (predicted == 3):
        return "OTHER"


def autoColorSelectHL(hue_value, light_value, model):
    inputData = [[hue_value / 180], [light_value / 255]]
    inputArray = np.array(inputData, dtype=np.float32)
    floatTensor = torch.from_numpy(inputArray)
    # Axis correction
    inputTensor = floatTensor.view(2)
    with torch.no_grad():
        outputs = model(inputTensor)
    return autoColorPicker(outputs)


def manuelColorSelect(hue_value, light_value):
    if (light_value > 229):
        return "WHITE"
    elif (hue_value < 5):
        return "RED"
    elif (hue_value < 22):
        return "ORANGE"
    elif (hue_value < 33):
        return "YELLOW"
    elif (hue_value < 78):
        return "GREEN"
    elif (hue_value < 131):
        return "BLUE"
    elif (hue_value < 170):
        return "VIOLET"
    else:
        return "RED"


def showImage(img, model):
    height, width, _ = img.shape
    while (1):
        cv2.imshow('image', img)

        hls_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        current_pixel = hls_frame[mouseY, mouseX]
        hue_value = current_pixel[0]
        light_value = current_pixel[1]

        manuelColor = manuelColorSelect(hue_value, light_value)
        autoColor = autoColorSelectHL(hue_value, light_value, model)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        current_pixel_bgr = img[mouseY, mouseX]
        b, g, r = int(current_pixel_bgr[0]), int(current_pixel_bgr[1]), int(current_pixel_bgr[2])

        cv2.rectangle(img, (0, 0), (110, 40), (0, 0, 0), -1)
        cv2.putText(img, autoColor, (0, 30), 0, 1, (b, g, r), 2)

        cv2.rectangle(img, (width-130, 0), (width, 40), (0, 0, 0), -1)
        cv2.putText(img, manuelColor, (width-130, 30), 0, 1, (b, g, r), 2)


if __name__ == "__main__":
    setup()
