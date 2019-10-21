from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import time
from collections import OrderedDict

import cv2
import dlib
import numpy as np
from scipy.spatial import distance


class ObjectTracker():
    def __init__(self, max_disappeared=50):
        self.next_obj_id = 0
        self.objects = OrderedDict()
        self.rects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, rect):
        self.objects[self.next_obj_id] = centroid
        self.rects[self.next_obj_id] = rect
        self.disappeared[self.next_obj_id] = 0
        self.next_obj_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.rects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            return self.objects, self.rects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_rects = np.zeros((len(rects), 4), dtype="int")

        for i, (start_x, start_y, end_x, end_y) in enumerate(rects):
            centroid_x = int((start_x + end_x) / 2.0)
            centroid_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (centroid_x, centroid_y)
            input_rects[i] = (start_x, start_y, end_x, end_y)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_rects[i])
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            dist = distance.cdist(np.array(obj_centroids), input_centroids)
            rows = dist.min(axis=1).argsort()
            cols = dist.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if (row in used_rows) or (col in used_cols):
                    continue

                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.rects[obj_id] = input_rects[col]
                self.disappeared[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(dist.shape[0])).difference(used_rows)
            unused_cols = set(range(dist.shape[1])).difference(used_cols)

            if dist.shape[0] >= dist.shape[1]:
                for row in unused_rows:
                    obj_id = obj_ids[row]
                    self.disappeared[obj_id] += 1

                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], input_rects[col])

        return self.objects, self.rects


def face_detector(model):
    realpath = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(realpath, "models", model)

    faces = dlib.get_frontal_face_detector()
    landmarks = dlib.shape_predictor(model_path)

    return faces, landmarks


def landmarks_detector(img, face, detector):
    rect = dlib.rectangle(
        int(face.left()),
        int(face.top()),
        int(face.right()),
        int(face.bottom()),
    )
    landmarks = detector(img, rect)

    return landmarks, [int(face.left()), int(face.top()), int(face.right()), int(face.bottom())]


def eye_aspect_ratio(points):
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])
    EAR = (A + B) / (2.0 * C)

    return EAR


def play(img, player_face_rect, landmarks, **info_eye):
    points, rect = landmarks_detector(img, player_face_rect, landmarks)
    ear = check_blink(points, **info_eye)

    return ear, rect


def shape_eye(landmarks, position):
    result = []
    for i in range(len(position)):
        x = landmarks.parts()[position[i]].x
        y = landmarks.parts()[position[i]].y
        result.append((x, y))

    return result


def check_blink(landmarks, left_eye, right_eye):
    shape_left_eye = shape_eye(landmarks, left_eye)
    shape_right_eye = shape_eye(landmarks, right_eye)
    
    left_ear = eye_aspect_ratio(shape_left_eye)
    right_ear = eye_aspect_ratio(shape_right_eye)

    ear = (left_ear + right_ear) / 2.0
    
    return ear


def run(args):
    if args.test is True:
        time.sleep(1)
        return
    else:
        cap = cv2.VideoCapture(args.cam_id)

    faces, landmarks = face_detector(args.model)
    ot = ObjectTracker()

    info_eye = {
        'left_eye': list(range(36, 42, 1)),
        'right_eye': list(range(42, 48, 1)),
    }

    previous_time = 0

    if (cap.isOpened() is False):
        print("Error opening video stream.")
    else:
        print("Press 'q' if you want to quit.")

    player1_blink = False
    player2_blink = False
    winner = ""
    game = OrderedDict()
    announce = ""

    while(cap.isOpened()):
        ret, img = cap.read()
        current_time = time.time()

        if ret is True:
            flipped_img = cv2.flip(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 1)
            face_rectangles = faces(flipped_img, 0)
            # H, W, C = flipped_img.shape

            if len(face_rectangles) == 2:
                rects = []
                players_ear = []
                dummy_rect = []
                for i, face in enumerate(face_rectangles):
                    rect = [int(face.left()), int(face.top()), int(face.right()), int(face.bottom())]
                    rects.append(rect)

                    ear, _rect = play(flipped_img, face, landmarks, **info_eye)
                    players_ear.append(ear)
                    dummy_rect.append(_rect)

                objects, rectangles = ot.update(rects)
                
                for obj_id, rect in rectangles.items():
                    cv2.rectangle(
                        flipped_img, 
                        (rect[0], rect[1]), 
                        (rect[2], rect[3]), 
                        (255 * obj_id, 255 * ((obj_id - 1) % 2), 0), 
                        1
                    )
                    for ear, _rect in zip(players_ear, dummy_rect):
                        if (_rect[0] == rect[0]) and (_rect[1] == rect[1]) and (_rect[2] == rect[2]) and (_rect[3] == rect[3]):
                            text = "Player {}".format(obj_id + 1)
                            cv2.putText(
                                flipped_img, text, 
                                (rect[0] - 10, rect[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255 * obj_id, 255 * ((obj_id - 1) % 2), 0), 
                                1
                            )
                            game[obj_id] = ear
                        else:
                            pass
                
                players = list(game.keys())
                if game[players[0]] < args.ratio:
                    player1_blink = True
                else:
                    player1_blink = False
                if game[players[1]] < args.ratio:
                    player2_blink = True
                else:
                    player2_blink = False

                if (player1_blink is False) and (player2_blink is True):
                    winner = "Winner is the Player1"
                    judge = "[INFO] " + winner
                    judge_time = time.time()
                    if announce != judge:
                        announce = judge
                        print(announce)
                elif (player1_blink is True) and (player2_blink is False):
                    winner = "Winner is the Player2"
                    judge = "[INFO] " + winner
                    judge_time = time.time()
                    if announce != judge:
                        announce = judge
                        print(announce)
            else:
                judge = "[Warning] Need only 2 players"
                judge_time = time.time() - 4
                if announce != judge:
                    announce = judge
                    print(announce)

            # update time
            diff_time = current_time - previous_time
            previous_time = current_time
            fps = "FPS: {:.1f}".format(1 / diff_time)

            cv2.putText(flipped_img, fps, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(flipped_img, winner, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.imshow("Staring Contests", flipped_img)

            if time.time() - judge_time > 3:
                winner = ""

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="shape_predictor_68_face_landmarks.dat", help="dlib detector model")
    parser.add_argument("-c", "--cam_id", type=int, default=0, help="webcam ID")
    parser.add_argument("-r", "--ratio", type=float, default=0.2, help="EAR threshold")
    parser.add_argument("-t", "--test", action="store_true", help="to interrupt for testing")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    run(args)


if __name__ == '__main__':
    main()
