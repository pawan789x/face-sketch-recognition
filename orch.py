import cv2
import phase_zero

I = cv2.imread("./data/sketch-small.png", cv2.IMREAD_GRAYSCALE)

I = phase_zero.align_face(I)
phase_zero.compute_gradient(I)

cv2.imshow('image', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
