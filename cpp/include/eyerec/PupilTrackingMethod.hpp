#ifndef CPP_INCLUDE_EYEREC_PUPILTRACKINGMETHOD_H
#define CPP_INCLUDE_EYEREC_PUPILTRACKINGMETHOD_H

#include <deque>
#include <memory>
#include <string>
#include <utility>

#include "opencv2/core.hpp"

#include "eyerec/PupilDetectionMethod.hpp"

struct TrackingParameters : public DetectionParameters {
    // Maximum time to keep found pupils in tracking buffer
    Timestamp maxAge = 300;
    // Minimum confidence required to track a pupil
    float minDetectionConfidence = 0.7f;
};

class PupilTrackingMethod {
public:
    virtual ~PupilTrackingMethod() = default;

    // Tracking and detection logic
    virtual Pupil detectAndTrack(
        const Timestamp& ts, const cv::Mat& frame, TrackingParameters params);

    virtual std::string description() = 0;

    virtual std::shared_ptr<PupilDetectionMethod> defaultPupilDetectionMethod()
        = 0;

    void setPupilDetectionMethod(std::shared_ptr<PupilDetectionMethod> method)
    {
        pupilDetectionMethod = method;
    }

protected:
    cv::Size expectedFrameSize = { 0, 0 };
    TrackedPupil previousPupil;
    std::deque<TrackedPupil> previousPupils;

    Timestamp maxAge = 300;
    float minDetectionConfidence = 0.7f;

    void registerPupil(const Timestamp& ts, Pupil& pupil);

    void reset();

    std::shared_ptr<PupilDetectionMethod> pupilDetectionMethod = nullptr;
    Pupil detect(const cv::Mat& frame, DetectionParameters params);

private:
    // Tracking last pupil implementation
    virtual void track(const cv::Mat& frame, const Pupil& previousPupil,
        Pupil& pupil, TrackingParameters params)
        = 0;

    TrackingParameters estimateTemporalROI(
        const Timestamp& ts, const TrackingParameters& oldParams);
};

#endif // CPP_INCLUDE_EYEREC_PUPILTRACKINGMETHOD_H
