import cv2
import phase_zero
import stasm as asm

print(asm.search_single)

I = cv2.imread("./data/photos_cropped/f-019-01.jpg", cv2.IMREAD_GRAYSCALE)
# I = cv2.imread("./data/sketch-small.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original', I)

I = phase_zero.align_face(I)

I1 = cv2.medianBlur(I, 5)
I2 = cv2.bilateralFilter(I, 4, 75, 75)

cv2.imshow('Gradient', phase_zero.compute_gradient(I1))

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(I,None)
img = cv2.drawKeypoints(I, kp, I)
cv2.imshow('Image SIFT', img)
landmarks = phase_zero.apply_stasm(I1)

# for point in landmarks:
#     cv2.imshow('stasm-out', cv2.circle(I, (point[0], point[1]), 3, 255, -1))

eyeLeft, eyeRight, nose, mouth, face = phase_zero.segment_stasm_data(I, landmarks)

cv2.imshow('eyeRight', eyeRight)
cv2.imshow('eyeLeft', eyeLeft)
cv2.imshow('nose', nose)
cv2.imshow('mouth', mouth)
cv2.imshow('face', face)

cv2.waitKey(0)
cv2.destroyAllWindows()
