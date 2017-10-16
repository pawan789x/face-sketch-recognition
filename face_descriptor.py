import cv2

from stasm_face import StasmFace


class FaceDescriptor(StasmFace):
    def __init__(self, img):
        StasmFace.__init__(self, img)

        self.keypoints = {}
        self.descriptors = {}

        sift = cv2.xfeatures2d.SIFT_create()

        for name, segment in self.segments.items():
            self.keypoints[name], self.descriptors[name] = sift.detectAndCompute(segment, None)