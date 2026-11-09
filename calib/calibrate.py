import glob
import math
import os
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
OUT_FILE = os.path.join(HERE, "webcam_1280x720.yaml")

# AprilGrid geometry (measured). Lengths in metres, so poses come out in metres.
COLS, ROWS = 6, 6
TAG_SIZE = 0.0315   # outer edge of the black square, including border
TAG_GAP = 0.0095    # white gap between neighbouring tags
PITCH = TAG_SIZE + TAG_GAP
FOCUS = 260         # webcam focus_absolute used during capture
MIN_TAGS = 6        # minimum detected tags for an image to be used
EXCLUDE = []        # image file names to leave out, e.g. ["img_007.png"]


def tag_object_corners(tag_id):
    """3D corners (m) of one tag on the board, in the detector's corner order
    (the tag's own top-left, top-right, bottom-right, bottom-left).
    Board frame: origin at the bottom-left corner of tag 0, X right (towards
    ID+1), Y up (towards ID+6), Z = 0 on the board.
    On this board each tag is printed rotated 180 deg relative to the board
    frame (verified with check_orientation.py), so the tag's own top-left
    corner is the bottom-right corner in board coordinates."""
    x0 = (tag_id % COLS) * PITCH
    y0 = (tag_id // COLS) * PITCH
    return np.array([[x0 + TAG_SIZE, y0,            0],
                     [x0,            y0,            0],
                     [x0,            y0 + TAG_SIZE, 0],
                     [x0 + TAG_SIZE, y0 + TAG_SIZE, 0]], dtype=np.float32)


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.markerBorderBits = 2  # AprilGrid tags use a 2-cell black border
# Sub-pixel refinement locates each corner more precisely than a whole pixel.
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

obj_points, img_points, names = [], [], []
image_size = None

for path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png"))):
    name = os.path.basename(path)
    if name in EXCLUDE:
        continue
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    size = (gray.shape[1], gray.shape[0])  # (width, height)
    if image_size is None:
        image_size = size
    elif size != image_size:
        raise SystemExit("%s has a different size: %s" % (name, size))

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < MIN_TAGS:
        print("Skipping %s: too few tags" % name)
        continue

    obj, img = [], []
    for c, tag_id in zip(corners, ids.flatten()):
        tag_id = int(tag_id)
        if tag_id >= COLS * ROWS:
            continue  # ID not on this board
        obj.append(tag_object_corners(tag_id))
        img.append(c[0])
    obj_points.append(np.concatenate(obj))
    img_points.append(np.concatenate(img).astype(np.float32).reshape(-1, 1, 2))
    names.append(name)

if len(names) < 10:
    raise SystemExit("Only %d usable images; capture more" % len(names))
print("Using %d images" % len(names))

# Fits camera matrix and distortion coefficients (k1, k2, p1, p2, k3).
# Also returns the standard deviation of each estimate and the RMS error per image.
(rms, K, dist, rvecs, tvecs,
 std_int, std_ext, per_view) = cv2.calibrateCameraExtended(
    obj_points, img_points, image_size, None, None)

w, h = image_size
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
s = std_int.flatten()

print("\nRMS reprojection error: %.3f px" % rms)
print("fx = %.1f +/- %.1f px" % (fx, s[0]))
print("fy = %.1f +/- %.1f px" % (fy, s[1]))
print("cx = %.1f +/- %.1f px   (image centre %.1f)" % (cx, s[2], w / 2))
print("cy = %.1f +/- %.1f px   (image centre %.1f)" % (cy, s[3], h / 2))
print("Horizontal FOV: %.1f deg   Vertical FOV: %.1f deg" % (
    math.degrees(2 * math.atan(w / (2 * fx))),
    math.degrees(2 * math.atan(h / (2 * fy)))))
print("Previously assumed focal length (70 deg HFOV): %.1f px"
      % (w / (2 * math.tan(math.radians(35)))))
print("Distortion (k1, k2, p1, p2, k3):", np.round(dist.flatten(), 4))

# Per-image errors, worst first. Images well above the median are suspect.
errors = per_view.flatten()
median = float(np.median(errors))
print("\nPer-image RMS error (median %.3f px):" % median)
for i in np.argsort(-errors):
    flag = "  <-- check" if errors[i] > 2 * median else ""
    print("  %s  %.3f px%s" % (names[i], errors[i], flag))

fs = cv2.FileStorage(OUT_FILE, cv2.FILE_STORAGE_WRITE)
fs.write("calibration_date", time.strftime("%Y-%m-%d %H:%M"))
fs.write("image_width", int(w))
fs.write("image_height", int(h))
fs.write("focus_absolute", FOCUS)
fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)
fs.write("rms_reprojection_error", float(rms))
fs.write("num_images", len(names))
fs.write("board", "AprilGrid 6x6 tag36h11, 2-bit border, tag 31.5 mm, gap 9.5 mm")
fs.release()
print("\nSaved", OUT_FILE)
