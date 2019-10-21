import pytest


class Args():
    model = "shape_predictor_68_face_landmarks.dat"
    cam_id = 0
    ratio = 0.2
    test = True


@pytest.fixture()
def args():
    result = Args()

    return result
