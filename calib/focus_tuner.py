import subprocess
import cv2

DEVICE = "/dev/video0"
PIPELINE = ("v4l2src device=/dev/video0 ! image/jpeg, width=1280, height=720 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1")

STEP = 10
FOCUS_MIN, FOCUS_MAX = 0, 1023

# Key codes returned by cv2.waitKeyEx for the arrow keys (Linux GTK backend).
KEY_UP, KEY_DOWN = 65362, 65364


def set_ctrl(name, value):
    """Set a V4L2 camera control using v4l2-ctl."""
    subprocess.run(["v4l2-ctl", "-d", DEVICE,
                    "--set-ctrl=%s=%d" % (name, value)], check=True)


focus = 260
set_ctrl("focus_auto", 0)
set_ctrl("focus_absolute", focus)

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise SystemExit("Could not open camera with GStreamer pipeline")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.markerBorderBits = 2  # AprilGrid tags use a 2-cell black border
detector = cv2.aruco.ArucoDetector(dictionary, params)

print("Up/Down or W/S: focus +/-%d   Q or Esc: quit" % STEP)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    n = 0 if ids is None else len(ids)

    # Variance of the Laplacian: higher values indicate a sharper image.
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    if n > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    text = "Focus: %d   Tags: %d   Sharpness: %.0f" % (focus, n, sharpness)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2)
    cv2.imshow("focus tuner", frame)

    key = cv2.waitKeyEx(1)
    if key != -1:
        # Strip modifier flags (e.g. Num Lock) from the key code.
        key &= 0xFFFF
    if key in (KEY_UP, ord('w')):
        focus = min(focus + STEP, FOCUS_MAX)
        set_ctrl("focus_absolute", focus)
    elif key in (KEY_DOWN, ord('s')):
        focus = max(focus - STEP, FOCUS_MIN)
        set_ctrl("focus_absolute", focus)
    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("Final focus value:", focus)
