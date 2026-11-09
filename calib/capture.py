import os
import subprocess
import time
import cv2
import numpy as np

DEVICE = "/dev/video0"
PIPELINE = ("v4l2src device=/dev/video0 ! image/jpeg, width=1280, height=720 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1")
FOCUS = 260           # fixed focus value; calibration is only valid at this value
MIN_TAGS = 6          # minimum detected tags for a frame to be saved
STILL_PX = 2.0        # max mean corner movement between frames to count as still
NEW_VIEW_PX = 30.0    # min mean corner movement since last save to count as a new view
SAVE_INTERVAL = 1.0   # min seconds between saves
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def set_ctrl(name, value):
    """Set a V4L2 camera control using v4l2-ctl."""
    subprocess.run(["v4l2-ctl", "-d", DEVICE,
                    "--set-ctrl=%s=%d" % (name, value)], check=True)


def mean_motion(a, b):
    """Mean corner displacement (px) over tag IDs present in both dicts.
    Returns None if fewer than MIN_TAGS IDs are shared."""
    common = set(a) & set(b)
    if len(common) < MIN_TAGS:
        return None
    return float(np.mean([np.linalg.norm(a[i] - b[i], axis=1).mean()
                          for i in common]))


set_ctrl("focus_auto", 0)
set_ctrl("focus_absolute", FOCUS)

os.makedirs(OUT_DIR, exist_ok=True)
# Continue numbering after any images already saved, so none are overwritten.
saved = len([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise SystemExit("Could not open camera with GStreamer pipeline")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.markerBorderBits = 2  # AprilGrid tags use a 2-cell black border
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Coverage map: accumulates the corner positions of every saved frame.
coverage = np.zeros((720, 1280, 3), np.uint8)

prev = {}        # tag ID -> corners in the previous frame
last_saved = {}  # tag ID -> corners in the last saved frame
last_save_time = 0.0
auto = False

print("SPACE: toggle auto-capture   Q or Esc: quit")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed")
        break
    if frame.shape[:2] != (720, 1280):
        raise SystemExit("Unexpected frame size: %s" % (frame.shape[:2],))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    n = 0 if ids is None else len(ids)
    current = {int(i): c[0] for c, i in zip(corners, ids.flatten())} if n else {}

    # Movement since the previous frame (stillness check).
    motion = mean_motion(current, prev)
    still = motion is not None and motion < STILL_PX
    prev = current

    # Movement since the last saved frame (new-view check).
    change = mean_motion(current, last_saved)
    new_view = change is None or change > NEW_VIEW_PX

    now = time.time()
    if (auto and n >= MIN_TAGS and still and new_view
            and now - last_save_time >= SAVE_INTERVAL):
        path = os.path.join(OUT_DIR, "img_%03d.png" % saved)
        cv2.imwrite(path, frame)
        saved += 1
        last_saved = current
        last_save_time = now
        print("Saved %s (%d tags)" % (path, n))
        for c in corners:
            for x, y in c[0]:
                cv2.circle(coverage, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Overlays are drawn on a copy so saved images stay unmodified.
    display = frame.copy()
    if n > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
    display = cv2.add(display, coverage)
    status = "AUTO ON" if auto else "AUTO OFF"
    text = "%s   Saved: %d   Tags: %d   %s" % (
        status, saved, n, "STILL" if still else "moving")
    cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2)
    cv2.imshow("capture", display)

    key = cv2.waitKeyEx(1)
    if key != -1:
        # Strip modifier flags (e.g. Num Lock) from the key code.
        key &= 0xFFFF

    if key == 32:  # space bar
        auto = not auto
        print("Auto-capture", "on" if auto else "off")
    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("Total images saved:", saved)
