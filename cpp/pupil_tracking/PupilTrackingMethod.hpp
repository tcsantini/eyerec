#ifndef PUPILTRACKINGMETHOD_H
#define PUPILTRACKINGMETHOD_H

#include <deque>
#include <memory>
#include <string>

#include "opencv2/core.hpp"

#include "PupilDetectionMethod.hpp"
#include "utils.h"

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
    virtual void detectAndTrack(const Timestamp& ts, const cv::Mat& frame, const cv::Rect& roi, Pupil& pupil, const float& minPupilDiameterPx = -1, const float& maxPupilDiameterPx = -1);

    virtual std::string description() = 0;

    virtual std::shared_ptr<PupilDetectionMethod> defaultPupilDetectionMethod() = 0;

    void setPupilDetectionMethod(std::shared_ptr<PupilDetectionMethod> method) { pupilDetectionMethod = method; }

protected:
    cv::Size expectedFrameSize = { 0, 0 };
    TrackedPupil previousPupil;
    std::deque<TrackedPupil> previousPupils;

    Timestamp maxAge = 300;
    float minDetectionConfidence = 0.7f;

    void registerPupil(const Timestamp& ts, Pupil& pupil);

    void reset();

    std::shared_ptr<PupilDetectionMethod> pupilDetectionMethod = nullptr;
    Pupil detect(const cv::Mat& frame, const cv::Rect& roi, const float& minPupilDiameterPx, const float& maxPupilDiameterPx);

private:
    // Tracking implementation
    virtual void track(const cv::Mat& frame, const cv::Rect& roi, const Pupil& previousPupil, Pupil& pupil, const float& minPupilDiameterPx = -1, const float& maxPupilDiameterPx = -1) = 0;

    cv::Rect estimateTemporalROI(const Timestamp& ts, const cv::Rect& roi);
};

#endif // PUPILTRACKINGMETHOD_H
