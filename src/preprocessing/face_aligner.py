# src/preprocessing/face_aligner.py

import os
import yaml
import cv2
import dlib
import numpy as np

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PREDICTOR_PATH = CONFIG["paths"]["shape_predictor"]

class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=128, desired_face_height=None):
        if not os.path.exists(PREDICTOR_PATH):
            raise FileNotFoundError(f"shape_predictor_68.dat not found at {PREDICTOR_PATH}")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height or desired_face_width

    def align(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) == 0:
            return image

        rect = rects[0]
        shape = self.predictor(gray, rect)
        coords = np.zeros((68, 2), dtype="float")
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        left_eye_pts = coords[36:42]
        right_eye_pts = coords[42:48]

        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")

        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        dist = np.sqrt((dx**2) + (dy**2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.desired_face_width
        scale = desired_dist / dist

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        output = cv2.warpAffine(image, M, (self.desired_face_width, self.desired_face_height),
                                flags=cv2.INTER_CUBIC)
        return output
