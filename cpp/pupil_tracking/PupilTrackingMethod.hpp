#ifndef PUPILTRACKINGMETHOD_H
#define PUPILTRACKINGMETHOD_H

#include <deque>
#include <memory>
#include <string>

#include "opencv2/core.hpp"

#include "utils.h"
#include "PupilDetectionMethod.hpp"

class TrackedPupil : public Pupil {
public:
    TrackedPupil(const Timestamp& ts, const Pupil& pupil)
        : Pupil(pupil)
        , ts(ts)
    {
    }

    TrackedPupil()
        : TrackedPupil(0, Pupil())
    {
    }

    Timestamp ts;
};

class PupilTrackingMethod {
public:
    virtual ~PupilTrackingMethod() = default;

    // Tracking and detection logic
    virtual void detectAndTrack(const Timestamp& ts, const cv::Mat& frame, const cv::Rect& roi, Pupil& pupil, std::shared_ptr<PupilDetectionMethod> pupilDetectionMethod = nullptr, const float& minPupilDiameterPx = -1, const float& maxPupilDiameterPx = -1);

    virtual std::string description() = 0;

protected:
    cv::Size expectedFrameSize = { 0, 0 };
    TrackedPupil previousPupil;
    std::deque<TrackedPupil> previousPupils;

    Timestamp maxAge = 300;
    float minDetectionConfidence = 0.7f;

    void registerPupil(const Timestamp& ts, Pupil& pupil);

    void reset();

private:
    // Detection implementation (can be overridden by providing a valid pointer to the detectAndTrack method, allowing the user to mix detectors and trackers)
    virtual Pupil detect(const cv::Mat& frame, const cv::Rect& roi, const float& minPupilDiameterPx = -1, const float& maxPupilDiameterPx = -1)
    {
        (void)frame;
        (void)roi;
        (void)minPupilDiameterPx;
        (void)maxPupilDiameterPx;
        return Pupil();
    }

    // Tracking implementation
    virtual void track(const cv::Mat& frame, const cv::Rect& roi, const Pupil& previousPupil, Pupil& pupil, const float& minPupilDiameterPx = -1, const float& maxPupilDiameterPx = -1) = 0;

    Pupil invokeDetection(const cv::Mat& frame, const cv::Rect& roi, std::shared_ptr<PupilDetectionMethod> pupilDetectionMethod, const float& minPupilDiameterPx, const float& maxPupilDiameterPx);
    cv::Rect estimateTemporalROI(const Timestamp& ts, const cv::Rect& roi);
};

#endif // PUPILTRACKINGMETHOD_H
