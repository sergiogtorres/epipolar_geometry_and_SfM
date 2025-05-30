import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def detect_and_match_keypoints(img1, img2):

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    best_matches = sorted(matches, key=lambda m: m.distance)[:20]

    good_matches = []
    pts1 = []
    pts2 = []

    for m1, m2 in best_matches:
        if m1.distance < 0.3 * m2.distance:
            good_matches.append(m1)
            pts1.append(kp1[m1.queryIdx].pt)
            pts2.append(kp2[m1.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    return pts1, pts2, kp1, kp2, good_matches

data_path = r"ignore/data_new/set_1"

img1_path = os.path.join(data_path, "img1.jpg")
img2_path = os.path.join(data_path, "img2.jpg")
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

pts1, pts2, kp1, kp2, good_matches = detect_and_match_keypoints(img1, img2)


img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
plt.imshow(img_matches)
plt.axis('off')
plt.show()

with open(os.path.join(data_path, "keypoints_1.txt"), "w") as file:
    N = len(pts1)
    file.write(str(N))
    for point in pts1:
        x, y = point
        file.write("\n" + str(x) + " " + str(y))

with open(os.path.join(data_path, "keypoints_2.txt"), "w") as file:
    for point in pts2:
        x, y = point
        file.write(str(x) + " " + str(y) + "\n")
