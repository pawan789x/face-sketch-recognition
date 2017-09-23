import itertools as it
import cv2
import stasm as asm
import math
import numpy as np

np.set_printoptions(threshold=np.nan)

# Face alignment using eye detection
# Refer to: https://stackoverflow.com/questions/39296461/how-to-straighten-a-tilted-face-after-cropping

def align_face(img):

    img_h, img_w = img.shape

    # divide the face into two halves for finding the eyes
    img_left = img[:, 0: math.floor(img_w / 2)]
    img_right = img[:, math.ceil(img_w / 2):]

    eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
    eyes_left = eye_cascade.detectMultiScale(img_left)
    eyes_right = eye_cascade.detectMultiScale(img_right)

    assert len(eyes_left) == 1, "Left eye not detected"
    assert len(eyes_right) == 1, "Right eye not detected"

    eyeL_x, eyeL_y, eyeL_w, eyeL_h = eyes_left[0]
    eyeR_x, eyeR_y, eyeR_w, eyeR_h = eyes_right[0]

    # shift the positions to complete image space
    eyeR_x += math.floor(img_w / 2)

    # todo: for better detection, don't divide by two

    # cv2.rectangle(I, (eyeL_x, eyeL_y), (eyeL_x + eyeL_w, eyeL_y + eyeL_h), 255, 2)
    # cv2.rectangle(I, (eyeR_x, eyeR_y), (eyeR_x + eyeR_w, eyeR_y + eyeR_h), 255, 2)

    rad = math.atan((eyeR_y - eyeL_y) / (eyeR_x - eyeL_x))
    M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), math.degrees(rad), 1)

    return cv2.warpAffine(img, M, (img_w, img_h))


# Gradient of the Image

def compute_gradient(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    angle = np.add(np.rad2deg(np.arctan(np.divide(sobel_y, sobel_x))), 180)
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    s = np.ones(img.shape) * 255

    hsv_image = np.uint8(cv2.merge([angle, s, magnitude]))

    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # cv2.imshow('Gradient Angle Hue 2', out2)
    # cv2.imshow('Gradient Magnitude', magnitude2)
    # cv2.imshow('Grad x', sobel_x)
    # cv2.imshow('Grad y', sobel_y)
    # cv2.imshow('Grad phase', img_phase)
    return out

def apply_stasm(img):
    landmarks = asm.search_single(img)
    print(landmarks)
    print(len(landmarks))
    if len(landmarks) == 0:
        print("No face found in the image")
    else:
        landmarks = asm.force_points_into_image(landmarks, img)
        landmarks = np.uint8(landmarks)
    return landmarks

def get_bounds(points):
    return [np.min(points, axis=0), np.max(points, axis=0)]

def crop_image(img, bounds):
    return img[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]]

def segment_stasm_data(img, landmarks):
    eyeLeft = [landmarks[i] for i in it.chain(range(30, 39), range(16, 22))]
    eyeRight = [landmarks[i] for i in it.chain(range(39, 48), range(22, 29))]
    nose = [landmarks[i] for i in range(48, 59)]
    mouth = [landmarks[i] for i in range(59, 77)]
    face = [landmarks[i] for i in range(0, 16)]

    return [crop_image(img, get_bounds(i)) for i in [eyeLeft, eyeRight, nose, mouth, face]]