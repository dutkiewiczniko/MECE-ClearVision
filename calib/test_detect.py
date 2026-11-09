import cv2

# Same capture pipeline matching the tracker (1280x720 MJPEG via GStreamer)
# drop=true max-buffers=1: keep only the newest frame, discard older ones.
PIPELINE = ("v4l2src device=/dev/video0 ! image/jpeg, width=1280, height=720 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1")

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
	raise SystemExit("Could not open camera with GStreamer pipeline")

# Discard initial frames to allow auto-exposure to settle
for _ in range(15):
	cap.read()
ok, frame = cap.read()
cap.release()
if not ok:
	raise SystemExit("Could not read a frame")
print("Frame size:", frame.shape[1], "x", frame.shape[0])

# Marker detection operates on a greyscale image.
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
# Tags printed with a 2-cell-thick black border 
params.markerBorderBits = 2
detector = cv2.aruco.ArucoDetector(dictionary, params)

# corners: four pixel corners per detected tag
# ids: decoded tag IDs
# rejected: square candidtes that failed to decode as a valid tag
corners, ids, rejected = detector.detectMarkers(gray)

n = 0 if ids is None else len(ids)
print("Detected tags:", n, "  Rejected candidates:", len(rejected))

if n > 0:
    # List each tag ID with its centre position, sorted top-to-bottom then
    # left-to-right, to show the ID layout of the board.
    centres = [(int(c[0][:, 1].mean()), int(c[0][:, 0].mean()), int(i))
               for c, i in zip(corners, ids.flatten())]
    centres.sort()
    for y, x, i in centres:
        print("ID %2d at x=%4d y=%4d" % (i, x, y))
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

cv2.imwrite("test_detect.png", frame)
cv2.imshow("detection", frame)
print("Press any key in the image window to close")
cv2.waitKey(0)
