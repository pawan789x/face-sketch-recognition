#!/usr/bin/python3
import face_preprocessor

from face_matcher import FaceMatcher
from face_descriptor import FaceDescriptor
from os import listdir, path

import  cv2

sketch = cv2.imread("./data/sketches_cropped/F2-019-01-sz1.jpg", cv2.IMREAD_GRAYSCALE)
sketch = face_preprocessor.align_face(sketch)

photos_dir = "./data/photos_cropped"

scoreboard = {}

for file in listdir(photos_dir):
    print(file)
    try:
        photo = face_preprocessor.align_face(cv2.imread(path.join(photos_dir, file), cv2.IMREAD_GRAYSCALE))
        score = FaceMatcher().match(FaceDescriptor(photo), FaceDescriptor(sketch))
        print("Score: ", score)
        scoreboard[path.join(photos_dir, file)] = score
    except AssertionError as err:
        print(err)
    print("\n")

best_matches = sorted(scoreboard, key=scoreboard.get)

cv2.imshow("Sketch", sketch)
cv2.imshow("#1", cv2.imread(best_matches[0]))
cv2.imshow("#2", cv2.imread(best_matches[1]))
cv2.imshow("#3", cv2.imread(best_matches[2]))

cv2.waitKey(0)
cv2.destroyAllWindows()