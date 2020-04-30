from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    fig = plt.figure(0)
    fig.canvas.set_window_title('Cumulative-sum, red=before, green=after')
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')


    # Display the images
    fig = plt.figure(1)
    fig.canvas.set_window_title('Original image')
    plt.imshow(img)

    fig = plt.figure(2)
    fig.canvas.set_window_title('Histogram equalization image')
    plt.imshow(imgeq)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()

    img_lst, err_lst = quantizeImage(img, 8, 20)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])

    fig = plt.figure(0)
    fig.canvas.set_window_title('First iteration of image quantization')
    plt.imshow(img_lst[0])
    fig = plt.figure(1)
    fig.canvas.set_window_title('Last iteration of image quantization')
    plt.imshow(img_lst[-1])

    fig = plt.figure(2)
    fig.canvas.set_window_title('Error of image quantization')
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    print("ID:", myID())
    img_path = 'testImg2.png'

    # Basic read and display
    imDisplay(img_path, LOAD_GRAY_SCALE)
    imDisplay(img_path, LOAD_RGB)

    # Convert Color spaces
    img = imReadAndConvert(img_path, LOAD_RGB)
    yiq_img = transformRGB2YIQ(img)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(yiq_img)
    plt.show()

    # Image histEq
    histEqDemo(img_path, LOAD_GRAY_SCALE)
    histEqDemo(img_path, LOAD_RGB)

    # Image Quantization
    quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)

    # Gamma
    gammaDisplay(img_path, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
