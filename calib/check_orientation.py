import glob
import math
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = 6

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.markerBorderBits = 2
detector = cv2.aruco.ArucoDetector(dictionary, params)


def angle_between(a, b):
    """Signed angle in degrees from vector a to vector b."""
    return math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0],
                                   a[0] * b[0] + a[1] * b[1]))


for path in sorted(glob.glob(os.path.join(HERE, "images", "*.png")))[:8]:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        continue
    tags = {int(i): c[0] for c, i in zip(corners, ids.flatten())}

    # A tag with a right-hand neighbour (ID+1, same row) and one above (ID+6).
    for t in sorted(tags):
        if t % COLS != COLS - 1 and t + 1 in tags and t + COLS in tags:
            break
    else:
        continue

    centre = tags[t].mean(axis=0)
    board_right = tags[t + 1].mean(axis=0) - centre
    board_up = tags[t + COLS].mean(axis=0) - centre
    tag_right = tags[t][1] - tags[t][0]   # corner 0 -> corner 1
    tag_up = tags[t][0] - tags[t][3]      # corner 3 -> corner 0

    print("%s  tag %2d   right: %7.1f deg   up: %7.1f deg" % (
        os.path.basename(path), t,
        angle_between(board_right, tag_right),
        angle_between(board_up, tag_up)))
