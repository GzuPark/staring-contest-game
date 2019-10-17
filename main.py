import cv2
import dlib
import numpy as np


def main():
    print("Check the packages version")
    print("{0: <5}: {1}".format('cv2', cv2.__version__))
    print("{0: <5}: {1}".format('dlib', dlib.__version__))
    print("{0: <5}: {1}".format('numpy', np.__version__))


if __name__ == '__main__':
    main()
