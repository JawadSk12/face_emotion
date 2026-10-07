# src/utils/visualizer.py

import cv2

def draw_face_box(frame, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def put_label(frame, text, x, y, color=(50, 255, 50)):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2)
