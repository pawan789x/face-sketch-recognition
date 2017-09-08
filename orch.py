import cv2
import phase_zero
import stasm as asm

print(asm.search_single)

# I = cv2.imread("./data/00001.jpg", cv2.IMREAD_GRAYSCALE)
I = cv2.imread("./data/sketch-small.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original', I)

I = phase_zero.align_face(I)

I1 = cv2.medianBlur(I, 5)
cv2.imshow('image median', I1)
I2 = cv2.bilateralFilter(I, 4, 75, 75)
cv2.imshow('image bilateral', I2)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(I,None)
img=cv2.drawKeypoints(I, kp, I)

cv2.imshow('Image SIFT', img)

phase_zero.compute_gradient(I2)
cv2.imshow('Message', phase_zero.apply_stasm(I))

cv2.waitKey(0)
cv2.destroyAllWindows()
