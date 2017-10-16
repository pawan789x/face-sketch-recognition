import math
import cv2

class FaceMatcher:
    def __init__(self):
        pass

    def match_descriptors(self, descriptor_1, descriptor_2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor_1, descriptor_2, 2)

        good = []
        total = 0
        cMatches = 0

        for match_1, match_2 in matches:
            if match_1.distance < 0.75 * match_2.distance:
                good.append(match_1)
                cMatches += 1
                total += match_1.distance

        if cMatches == 0: return math.nan

        return total / cMatches


    def match(self, photo, sketch):
        overall_score = 0
        count = 0
        for name in ["face", "eye_left", "eye_right", "nose", "mouth"]:
            score = self.match_descriptors(photo.descriptors[name], sketch.descriptors[name])
            if not math.isnan(score):
                overall_score += score
                count += 1
            print("Similarity Factor: ", name, score)
        return overall_score / count
