from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from time import time

import cv2
import dlib
import numpy as np


def check_version():
    print("Check the packages version")
    print("{0: <5}: {1}".format('cv2', cv2.__version__))
    print("{0: <5}: {1}".format('dlib', dlib.__version__))
    print("{0: <5}: {1}".format('numpy', np.__version__))


def webcam(cam_id):
    cap = cv2.VideoCapture(cam_id)

    previous_time = 0

    if (cap.isOpened() is False):
        print("Error opening video stream.")
    else:
        print("Press 'q' if you want to quit.")

    while(cap.isOpened()):
        ret, img = cap.read()
        current_time = time()

        if ret is True:
            flipped_img = cv2.flip(img, 1)

            # update time
            diff_time = current_time - previous_time
            previous_time = current_time
            fps = "FPS: {:.1f}".format(1 / diff_time)

            cv2.putText(flipped_img, fps, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow("Staring Contests", flipped_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cam_id", type=int, default=0, help="Webcam ID.")

    args = parser.parse_args()

    return args


def main():
    check_version()

    args = get_args()
    webcam(args.cam_id)


if __name__ == '__main__':
    main()
