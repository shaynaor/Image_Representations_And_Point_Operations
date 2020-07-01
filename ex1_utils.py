"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 000000


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns and returns in converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # read image:
    try:
        img = cv2.imread(filename)

    except cv2.error as e:
        print("cv2.error: ", e)
        exit(1)

    if img is None:
        raise Exception("Failed to read the image!")

    # convert the image into a given representation
    # (imread stored images in BGR order by default)
    if representation is LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation is LOAD_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("There are two possible inputs for representation- Grayscale or RGB")

    # normalize image to the range [0, 1] and return it
    return img / 255


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)  # display the image
    # if representation is LOAD_GRAY_SCALE add a gray flag to plt.
    if representation is LOAD_GRAY_SCALE:
        plt.gray()
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # input checking
    if imgRGB is None:
        raise Exception("Error: imgRGB is None!")
    # define the transformation matrix
    matTransRGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    imgYIQ = imgRGB.dot(matTransRGB2YIQ.T)
    return imgYIQ.copy()


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # input checking
    if imgYIQ is None:
        raise Exception("Error: imgYIQ is None!")
    # define the transformation matrix
    matTransYIQ2RGB = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]])
    imgRGB = imgYIQ.dot(matTransYIQ2RGB.T)
    # bounds checking:
    imgRGB[imgRGB < float(0)] = float(0)
    imgRGB[imgRGB > float(1)] = float(1)
    return imgRGB.copy()


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: is the input grayscale or RGB image to be equalized having values in the range [0, 1].
    :return: (imgEq,histOrg,histEQ)
    """
    # input checking
    if imgOrig is None:
        raise Exception("Error: imgOrig is None!")
    # handle RGB images
    flagRGB = False
    if len(imgOrig.shape) is 3:  # RGB image
        flagRGB = True
        imgYIQ = transformRGB2YIQ(imgOrig)  # transform to YIQ color space
        imgOrig = imgYIQ[:, :, 0]

    imgOrigInt = (imgOrig * 255).astype("uint8")
    # find the histogram of the original image
    histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))
    cumsumOrig = np.cumsum(histOrig)  # calculate cumulative-sum of the original image
    cdfNorm = (cumsumOrig * 255 / cumsumOrig[-1]).astype("uint8")  # normalize cumulative histogram
    imgEq = cdfNorm[imgOrigInt]  # apply the transformation
    # bounds checking:
    imgEq[imgEq < 0] = 0
    imgEq[imgEq > 255] = 255
    # calculate cumulative-sum of the new image
    histEq, _ = np.histogram(imgEq.flatten(), 256, range=(0, 255))

    # normalize image into range [0,1]
    imgEq = imgEq / 255

    if flagRGB:  # RGB image
        imgYIQ[:, :, 0] = imgEq.copy()  # modify y channel
        imgEq = transformYIQ2RGB(imgYIQ)  # transform back to RGB
        return imgEq, histOrig, histEq

    return imgEq, histOrig, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # input checking
    if imOrig is None:
        raise Exception("Error: imOrig is None!")
    if nQuant > 256:
        raise ValueError("nQuant is greater then 256!")
    if nIter < 0:
        raise ValueError("Number of optimization loops must be a positive number!")

    # handle RGB images
    flagRGB = False
    if len(imOrig.shape) is 3:  # RGB image
        flagRGB = True
        imgYIQ = transformRGB2YIQ(imOrig)  # transform to YIQ color space
        imOrig = imgYIQ[:, :, 0]  # y-channel

    imgOrigInt = (imOrig * 255).astype("uint8")
    # find the histogram of the original image
    histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))

    errors = []  # errors array
    encodedImages = []  # contains all the encoded images- finally return the minimum error image
    global intensities, z, q

    for j in range(nIter):
        encodeImg = imgOrigInt.copy()
        # Finding z - the values that each of the segments intensities will map to.
        if j is 0:  # first iteration INIT z
            z = np.arange(0, 255 - int(256 / nQuant) + 1, int(256 / nQuant))
            z = np.append(z, 255)
            intensities = np.array(range(256))
        else:  # not the first iteration
            for r in range(1, len(z) - 2):
                new_z_r = int((q[r - 1] + q[r]) / 2)
                if new_z_r != z[r - 1] and new_z_r != z[r + 1]:  # to avoid division by 0
                    z[r] = new_z_r

        # Finding q - the values that each of the segments intensities will map to.
        q = np.array([], dtype=np.float64)
        for i in range(len(z) - 1):
            mask = np.logical_and((z[i] < encodeImg), (encodeImg < z[i + 1]))  # the current cluster
            if i is not (len(z) - 2):
                # calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]]))
                else:  # to avoid division by 0
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]] + 0.001))
                encodeImg[mask] = int(q[i])  # apply the changes to the encoded image

            else:  # i is len(z)-2 , add 255
                # calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1], weights=histOrig[z[i]:z[i + 1] + 1]))
                else:  # to avoid division by 0
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1],
                                                weights=histOrig[z[i]:z[i + 1] + 1] + 0.001))
                encodeImg[mask] = int(q[i])  # apply the changes on the encoded image

        errors.append((np.square(np.subtract(imgOrigInt, encodeImg))).mean())  # calculate error
        encodeImg = encodeImg / 255  # normalize to range [0,1]

        if flagRGB:  # RGB image
            imgYIQ[:, :, 0] = encodeImg.copy()  # modify y channel
            encodeImg = transformYIQ2RGB(imgYIQ)  # transform back to RGB

        encodedImages.append(encodeImg)

        # checking whether we have come to convergence
        if j > 1 and abs(errors[j - 1] - errors[j]) < 0.01:
            print("we have come to convergence after {} iterations!".format(j + 1))
            break

    return encodedImages, errors
