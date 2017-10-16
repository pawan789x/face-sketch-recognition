import itertools as it

import numpy as np
import stasm as asm


def get_bounds(points):
    return [np.min(points, axis=0), np.max(points, axis=0)]


def crop_image(img, bounds):
    return img[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]]


class StasmFace:
    def __init__(self, img):
        self.landmarks = {}
        self.segments = {}

        # STASM
        landmarks = asm.search_single(img)

        assert len(landmarks) > 0, "cannot detect face"

        landmarks = asm.force_points_into_image(landmarks, img)
        landmarks = np.uint8(landmarks)

        self.segment_face(img, landmarks)

    def segment_face(self, img, landmarks):
        self.landmarks["face"] = [landmarks[i] for i in range(0, 16)]
        self.landmarks["eye_left"] = [landmarks[i] for i in it.chain(range(30, 39), range(16, 22))]
        self.landmarks["eye_right"] = [landmarks[i] for i in it.chain(range(39, 48), range(22, 29))]
        self.landmarks["nose"] = [landmarks[i] for i in range(48, 59)]
        self.landmarks["mouth"] = [landmarks[i] for i in range(59, 77)]

        self.segments["face"] = crop_image(img, get_bounds(self.landmarks["face"]))
        self.segments["eye_left"] = crop_image(img, get_bounds(self.landmarks["eye_left"]))
        self.segments["eye_right"] = crop_image(img, get_bounds(self.landmarks["eye_right"]))
        self.segments["nose"] = crop_image(img, get_bounds(self.landmarks["nose"]))
        self.segments["mouth"] = crop_image(img, get_bounds(self.landmarks["mouth"]))
        pass
