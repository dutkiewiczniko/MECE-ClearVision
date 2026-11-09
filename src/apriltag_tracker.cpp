#include "apriltag_tracker.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/videoio.hpp>

/* ---- Settings ---- */

// Webcam calibration (camera matrix, distortion, focus), valid only at 1280x720.
static const char *CALIB_REL_PATH = "/MECE-ClearVision/calib/webcam_1280x720.yaml";
static constexpr int CALIB_W = 1280, CALIB_H = 720;

// Phone tag: tag36h11 ID 100, printed black square measured as 46.0 mm.
static constexpr int   TAG_ID     = 100;
static constexpr float TAG_SIZE_M = 0.046f;

/* Phone rear (1x) camera lens position relative to the tag centre, in the
 * tag frame (X right, Y up, Z out of the tag, as seen looking at the tag).
 * Measured from the back of the phone: lens 22 mm left and 65 mm up from the
 * tag centre, with the tag stuck rotated 90 deg clockwise. In the tag's own
 * axes this is 65 mm along -X and 22 mm along -Y. */
static constexpr float LENS_X = -0.065f;
static constexpr float LENS_Y = -0.022f;
static constexpr float LENS_Z =  0.000f;

// EMA smoothing factor: higher = more responsive, lower = steadier.
static constexpr float AT_SMOOTH = 0.5f;

/* ---- Public interface ---- */

bool AprilTagTracker::start(int camera_index, const std::string &)
{
    camera_index_ = camera_index;
    running_      = true;
    worker_       = std::thread(&AprilTagTracker::threadLoop, this);
    return true;
}

void AprilTagTracker::stop()
{
    running_ = false;
    if (worker_.joinable()) worker_.join();
}

HeadPos AprilTagTracker::headPos() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return head_pos_;
}

/* ---- Background thread ---- */

void AprilTagTracker::threadLoop()
{
    /* Load intrinsics and focus value from the calibration file. */
    const char *home = std::getenv("HOME");
    std::string calib_path = std::string(home ? home : "") + CALIB_REL_PATH;
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        fprintf(stderr, "[AprilTag] Cannot open %s\n", calib_path.c_str());
        running_ = false;
        return;
    }
    cv::Mat K, dist;
    int focus = 0;
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    fs["focus_absolute"] >> focus;
    fs.release();
    if (K.empty() || dist.empty()) {
        fprintf(stderr, "[AprilTag] Calibration file is missing values\n");
        running_ = false;
        return;
    }

    /* Intrinsics are only valid at the calibration focus, so lock it. */
    char cmd[256];
    std::snprintf(cmd, sizeof(cmd),
                  "v4l2-ctl -d /dev/video%d --set-ctrl=focus_auto=0 "
                  "--set-ctrl=focus_absolute=%d", camera_index_, focus);
    if (std::system(cmd) != 0)
        fprintf(stderr, "[AprilTag] Warning: could not set webcam focus\n");

    /* Open the webcam at the calibrated resolution only. */
    char pipeline[512];
    std::snprintf(pipeline, sizeof(pipeline),
                  "v4l2src device=/dev/video%d"
                  " ! image/jpeg, width=%d, height=%d"
                  " ! jpegdec ! videoconvert"
                  " ! video/x-raw, format=BGR"
                  " ! appsink drop=true max-buffers=1",
                  camera_index_, CALIB_W, CALIB_H);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        fprintf(stderr, "[AprilTag] Failed to open camera %d at %dx%d\n",
                camera_index_, CALIB_W, CALIB_H);
        running_ = false;
        return;
    }
    printf("[AprilTag] Camera %d: %dx%d, focus %d, tag ID %d (%.1f mm)\n",
           camera_index_, CALIB_W, CALIB_H, focus, TAG_ID, TAG_SIZE_M * 1000.f);

    /* Detector for standard (1-cell border) tag36h11 markers. */
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::ArucoDetector detector(
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11), params);

    /* Tag corners in the tag frame, in the order required by
     * SOLVEPNP_IPPE_SQUARE (matches the detector's corner order). */
    const float h = TAG_SIZE_M / 2.f;
    const std::vector<cv::Point3f> obj = {
        {-h,  h, 0.f}, { h,  h, 0.f}, { h, -h, 0.f}, {-h, -h, 0.f}};
    const cv::Mat lens_tag = (cv::Mat_<double>(3, 1) << LENS_X, LENS_Y, LENS_Z);

    cv::Mat frame, gray;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;

    while (running_) {
        if (!cap.read(frame) || frame.empty()) continue;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        detector.detectMarkers(gray, corners, ids);

        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] != TAG_ID) continue;

            cv::Mat rvec, tvec;
            if (!cv::solvePnP(obj, corners[i], K, dist, rvec, tvec,
                              false, cv::SOLVEPNP_IPPE_SQUARE))
                break;

            /* Lens position in the webcam frame: rotate the tag-frame
             * offset into the camera frame and add the tag position. */
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            cv::Mat lens_cam = R * lens_tag + tvec;
            const float x = (float)lens_cam.at<double>(0);
            const float y = (float)lens_cam.at<double>(1);
            const float z = (float)lens_cam.at<double>(2);

            std::lock_guard<std::mutex> lock(mtx_);
            if (!head_pos_.valid) {
                head_pos_.x = x; head_pos_.y = y; head_pos_.z = z;
            } else {
                head_pos_.x = AT_SMOOTH * x + (1.f - AT_SMOOTH) * head_pos_.x;
                head_pos_.y = AT_SMOOTH * y + (1.f - AT_SMOOTH) * head_pos_.y;
                head_pos_.z = AT_SMOOTH * z + (1.f - AT_SMOOTH) * head_pos_.z;
            }
            head_pos_.valid = true;   // last position is held if the tag is lost
            active_ = true;
            break;
        }
    }
    cap.release();
}
