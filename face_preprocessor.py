import math

import cv2

def align_face(img):
    img_h, img_w = img.shape

    # divide the face into two halves for finding the eyes
    img_left = img[:, 0: math.floor(img_w / 2)]
    img_right = img[:, math.ceil(img_w / 2):]

    # initialize the classifier
    eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')

    # detect the eyes
    eyes_left = eye_cascade.detectMultiScale(img_left)
    eyes_right = eye_cascade.detectMultiScale(img_right)

    assert len(eyes_left) == 1, "Left eye not detected"
    assert len(eyes_right) == 1, "Right eye not detected"

    eyeL_x, eyeL_y, eyeL_w, eyeL_h = eyes_left[0]
    eyeR_x, eyeR_y, eyeR_w, eyeR_h = eyes_right[0]

    # shift the positions to complete image space
    eyeR_x += math.floor(img_w / 2)

    # TODO for better detection, don't divide by two

    rad = math.atan((eyeR_y - eyeL_y) / (eyeR_x - eyeL_x))
    M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), math.degrees(rad), 1)

    return cv2.warpAffine(img, M, (img_w, img_h))
