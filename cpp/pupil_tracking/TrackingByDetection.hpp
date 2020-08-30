#ifndef TRACKINGBYDETECTION_H
#define TRACKINGBYDETECTION_H

#include "PupilTrackingMethod.hpp"

template <class T>
class TrackingByDetection : public PupilTrackingMethod {

public:
    void detectAndTrack(const Timestamp& ts, const cv::Mat& frame, const cv::Rect& roi, Pupil& pupil,
        const float& minPupilDiameterPx = -1, const float& maxPupilDiameterPx = -1) override
    {
        pupil = detect(frame, roi, minPupilDiameterPx, maxPupilDiameterPx);
    };
    void track(const cv::Mat& frame, const cv::Rect& roi, const Pupil& previousPupil, Pupil& pupil,
        const float& userMinPupilDiameterPx = -1, const float& userMaxPupilDiameterPx = -1){};
    std::string description()
    {
        if (!pupilDetectionMethod)
            setPupilDetectionMethod(defaultPupilDetectionMethod());
        return pupilDetectionMethod->description();
    }
    std::shared_ptr<PupilDetectionMethod> defaultPupilDetectionMethod() { return std::make_shared<T>(); }
};

#endif // TRACKINGBYDETECTION_H
