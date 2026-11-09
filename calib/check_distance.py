import math
import os
import subprocess
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "webcam_1280x720.yaml")
DEVICE = "/dev/video0"
PIPELINE = ("v4l2src device=/dev/video0 ! image/jpeg, width=1280, height=720 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1")

# AprilGrid geometry (measured), in metres.
COLS, ROWS = 6, 6
TAG_SIZE = 0.0315
TAG_GAP = 0.0095
PITCH = TAG_SIZE + TAG_GAP
MIN_TAGS = 4
# Centre of the tag area, in board coordinates.
BOARD_CENTRE = np.array([((COLS - 1) * PITCH + TAG_SIZE) / 2,
                         ((ROWS - 1) * PITCH + TAG_SIZE) / 2, 0.0])


def set_ctrl(name, value):
    """Set a V4L2 camera control using v4l2-ctl."""
    subprocess.run(["v4l2-ctl", "-d", DEVICE,
                    "--set-ctrl=%s=%d" % (name, value)], check=True)


def tag_object_corners(tag_id):
    """3D corners (m) of one tag, in detector corner order. Tags on this board
    are printed rotated 180 deg relative to the board frame (see calibrate.py)."""
    x0 = (tag_id % COLS) * PITCH
    y0 = (tag_id // COLS) * PITCH
    return np.array([[x0 + TAG_SIZE, y0,            0],
                     [x0,            y0,            0],
                     [x0,            y0 + TAG_SIZE, 0],
                     [x0 + TAG_SIZE, y0 + TAG_SIZE, 0]], dtype=np.float32)


def board_depth(obj, img, K, dist):
    """Pose of the board from solvePnP; returns (rvec, tvec, depth of board
    centre along the camera's optical axis in metres)."""
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    centre_cam = R @ BOARD_CENTRE + tvec.flatten()
    return rvec, tvec, centre_cam[2]


# Calibrated intrinsics from the YAML file.
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("distortion_coefficients").mat()
focus = int(fs.getNode("focus_absolute").real())
fs.release()

# Previously assumed intrinsics: 70 deg HFOV, image centre, no distortion.
f_assumed = 1280 / (2 * math.tan(math.radians(35)))
K_assumed = np.array([[f_assumed, 0, 640],
                      [0, f_assumed, 360],
                      [0, 0, 1]], dtype=np.float64)
dist_assumed = np.zeros(5)

set_ctrl("focus_auto", 0)
set_ctrl("focus_absolute", focus)

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise SystemExit("Could not open camera with GStreamer pipeline")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.markerBorderBits = 2  # AprilGrid tags use a 2-cell black border
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

print("Q or Esc: quit")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    lines = ["Tags: 0"]
    if ids is not None:
        obj, img = [], []
        for c, tag_id in zip(corners, ids.flatten()):
            if int(tag_id) < COLS * ROWS:
                obj.append(tag_object_corners(int(tag_id)))
                img.append(c[0])
        n = len(obj)
        lines = ["Tags: %d" % n]
        if n >= MIN_TAGS:
            obj = np.concatenate(obj)
            img = np.concatenate(img).astype(np.float32)
            calib = board_depth(obj, img, K, dist)
            assumed = board_depth(obj, img, K_assumed, dist_assumed)
            if calib is not None:
                rvec, tvec, z = calib
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.05)
                lines.append("Calibrated Z: %.3f m" % z)
            if assumed is not None:
                lines.append("Assumed Z:    %.3f m" % assumed[2])

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2)
    # Preview scaled to 80% to fit a 1080 px wide (portrait) screen.
    cv2.imshow("distance check", cv2.resize(frame, None, fx=0.8, fy=0.8))

    key = cv2.waitKeyEx(1)
    if key != -1:
        key &= 0xFFFF  # strip modifier flags (e.g. Num Lock)
    if key == ord('p'):
        print(" | ".join(lines))  # print current readings to the terminal
    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
