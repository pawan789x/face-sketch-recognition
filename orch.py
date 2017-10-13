import cv2
import numpy as np
import phase_zero
import stasm as asm

print(asm.search_single)

sift = cv2.xfeatures2d.SIFT_create()


def function(I, name):
    # I = cv2.imread("./data/sketch-small.png", cv2.IMREAD_GRAYSCALE)

    cv2.imshow('Original' + name, I)

    I = phase_zero.align_face(I)

    I1 = cv2.medianBlur(I, 5)
    I2 = cv2.bilateralFilter(I, 4, 75, 75)

    cv2.imshow('Gradient' + name, phase_zero.compute_gradient(I1))

    kp = sift.detect(I, None)
    img = cv2.drawKeypoints(I, kp, I)
    cv2.imshow('Image SIFT' + name, img)
    landmarks = phase_zero.apply_stasm(I1)

    # for point in landmarks:
    #     cv2.imshow('stasm-out', cv2.circle(I, (point[0], point[1]), 3, 255, -1))

    eyeLeft, eyeRight, nose, mouth, face = phase_zero.segment_stasm_data(I, landmarks)

    eyeRight_kp, eyeRight_img, eyeRight_desc = siftf(eyeRight)
    eyeLeft_kp, eyeLeft_img, eyeLeft_desc = siftf(eyeLeft)
    nose_kp, nose_img, nose_desc = siftf(nose)
    mouth_kp, mouth_img, mouth_desc = siftf(mouth)
    face_kp, face_img, face_desc = siftf(face)

    cv2.imshow('eyeRight' + name, eyeRight_img)
    cv2.imshow('eyeLeft' + name, eyeLeft_img)
    cv2.imshow('nose' + name, nose_img)
    cv2.imshow('mouth' + name, mouth_img)
    cv2.imshow('face' + name, face_img)

    return face_img, face_kp, face_desc


def siftf(I):
    kp = sift.detect(I, None)
    img = cv2.drawKeypoints(I, kp, I)
    kp, desc = sift.compute(I, kp)
    return kp, img, desc


I1 = cv2.imread("./data/photos_cropped/f-019-01.jpg", cv2.IMREAD_GRAYSCALE)
face_img, face_kp, face_desc = function(I1, ' original')
I2 = cv2.imread("./data/sketches_cropped/F2-019-01-sz1.jpg", cv2.IMREAD_GRAYSCALE)
face_sketch_img, face_kp_sketch, face_desc_sketch = function(I2, ' sketch')

bf = cv2.BFMatcher()
matches = bf.knnMatch(face_desc, face_desc_sketch, 2)

good = []
total = 0
cMatches = 0
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
        cMatches += 1
        total += m.distance

img3 = cv2.drawMatchesKnn(face_img, face_kp, face_sketch_img, face_kp_sketch, matches, None, flags=2)
print('Similarity factor:', total / cMatches)

cv2.imshow('new main', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
