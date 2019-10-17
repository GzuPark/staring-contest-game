from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import dlib


def face_detector(model):
    realpath = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(realpath, "models", model)

    faces = dlib.get_frontal_face_detector()
    landmarks = dlib.shape_predictor(model_path)

    return faces, landmarks
