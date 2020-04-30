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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np


def printGammaValues(gammaValue):
    print(gammaValue / 100)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # read image:
    img = cv2.imread(img_path)
    # set GUI name
    cv2.namedWindow("Gamma Correction", cv2.WINDOW_KEEPRATIO)
    # create track-bar
    cv2.createTrackbar("Gamma", "Gamma Correction", 100, 200, printGammaValues)

    while True:
        # gets the track-bar position
        gamma = cv2.getTrackbarPos("Gamma", "Gamma Correction")

        # apply gamma power-low transform (divide by 100 to match gamma GUI values)
        gammaCorrectedImg = np.array((img / 255) ** (gamma / 100), dtype='float64')

        cv2.imshow("Gamma Correction", gammaCorrectedImg)

        # display the frame to 1 ms, if 'q' press-break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    gammaDisplay('gamma_example-g10b.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
