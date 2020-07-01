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
IMG_INT_MAX_VAL = 255


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


def recolor(img: np.ndarray, borders: np.ndarray, weighted_mean: np.ndarray) -> np.ndarray:
    """
    Recolors the image according to the borders and the cells's means
    :param img: The original image
    :param borders: The borders
    :param weighted_mean: The cells's means
    :return: The recolored image
    """
    img_quant = np.zeros_like(img)
    for q in range(len(borders) - 1):
        left_b = borders[q]
        right_b = borders[q + 1]
        min_val_matrix = img >= left_b
        max_val_matrix = img < right_b
        img_quant[max_val_matrix * min_val_matrix] = weighted_mean[q]
    return img_quant


def calHist(img: np.ndarray, bins=256, hist_range: tuple = (0, 256)) -> np.ndarray:
    """
    Calculate the image histogram
    :param img: Input image
    :param bins: Number of bins to group the values
    :param hist_range: The range of values
    :return: An np.array of size (bins)
    """
    img_flat = img.ravel()
    min_val = hist_range[0]
    max_val = hist_range[1]
    hist = np.zeros(bins)

    for pix in img_flat:
        quan_val = int(bins * (pix - min_val) / (max_val - min_val))
        hist[quan_val] += 1

    return hist


def calCumSum(arr: np.array) -> np.ndarray:
    """
    Calculate the Cumulitive Sum of an array
    :param arr: Input array
    :return: A CumSum array of size (len(arr))
    """
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)
    for idx in range(1, arr_len):
        cum_sum[idx] = arr[idx] + cum_sum[idx - 1]

    return cum_sum


def getWeightedMean(vals: np.ndarray, weights: np.ndarray) -> int:
    """
    Calculats the weighted mean of
    :param vals:
    :param weights:
    :return:
    """
    val = (vals * weights).sum() / (weights.sum() + np.finfo('float').eps)
    return val.astype(int)


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    gray = imOrig
    if len(imOrig.shape) > 2:
        yiq = transformRGB2YIQ(imOrig)
        gray = yiq[:, :, 0]
        comp_img = imOrig
    else:
        comp_img = (gray * IMG_INT_MAX_VAL).astype(np.int)
    gray = (gray * IMG_INT_MAX_VAL).astype(np.int)

    hist = calHist(gray)
    cumsum = calCumSum(hist)
    cumsum /= cumsum[-1]

    # Initiating the borders
    borders = np.arange(gray.min(), gray.max(), gray.max() / nQuant, dtype=np.float)
    max_val = gray.max() + 1
    borders = np.hstack([borders, max_val])
    weighted_mean = np.array([borders[x:x + 2].mean() for x in range(len(borders[:-1]))])

    # Calculate histogram
    error_trk = []
    img_quant_trk = []
    # dispBorders(hist, borders, weighted_mean, 100)
    for it in range(nIter):
        # Find quants means
        for q in range(nQuant):
            left_b = int(np.floor(borders[q]))
            right_b = int(np.ceil(borders[q + 1]))
            q_vals = hist[left_b:right_b]
            weighted_mean[q] = getWeightedMean(np.arange(left_b, right_b), q_vals)

        # Make shift borders
        borders = np.zeros_like(borders)
        borders[-1] = max_val
        for q in range(1, nQuant):
            borders[q] = weighted_mean[q - 1:q + 1].sum() / 2

        # Recolor the image
        tmp = recolor(gray, borders, weighted_mean)
        if len(imOrig.shape) > 2:
            tmp_yiq = yiq.copy()
            tmp_yiq[:, :, 0] = tmp / IMG_INT_MAX_VAL
            tmp = transformYIQ2RGB(tmp_yiq)
            tmp[tmp > 1] = 1
            tmp[tmp < 0] = 0
        img_quant_trk.append(tmp)
        error_trk.append(np.power(comp_img - tmp, 2).mean())

    return img_quant_trk, error_trk
