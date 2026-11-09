#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include "face_tracker.h"   // HeadPos

/*
 * AprilTagTracker: background thread that detects a single tag36h11 marker
 * (on the back of a phone) with the driver-facing webcam, estimates its pose
 * with solvePnP using the measured webcam intrinsics, and reports the position
 * of the phone's rear camera lens as the viewer's eye position.
 *
 * Output uses the same HeadPos convention as FaceTracker, so it can replace it
 * without changing the head-position mapping in view_synthesis.cpp:
 *   x: camera's right, y: down, z: away from the camera (metres).
 * Unlike FaceTracker, x/y are absolute (no calibration reference).
 */
class AprilTagTracker {
public:
    AprilTagTracker()  = default;
    ~AprilTagTracker() { stop(); }

    // Second argument is unused; kept so the call matches FaceTracker::start.
    bool start(int camera_index, const std::string &unused = "");
    void stop();

    // No-op: tag positions are absolute, so no reference is needed.
    void calibrate() {}

    HeadPos headPos() const;
    bool isActive() const { return active_.load(); }

private:
    void threadLoop();

    std::thread        worker_;
    std::atomic<bool>  running_{false};
    std::atomic<bool>  active_{false};
    mutable std::mutex mtx_;

    HeadPos head_pos_;
    int camera_index_{0};
};
