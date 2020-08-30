#ifndef TRACKINGBYDETECTION_H
#define TRACKINGBYDETECTION_H

#include "PupilTrackingMethod.hpp"

template <class T>
class TrackingByDetection : public PupilTrackingMethod {

public:
    void detectAndTrack(const Timestamp& ts, const cv::Mat& frame, Pupil& pupil,
        TrackingParameters params) override
    {
        track(frame, Pupil(), pupil, params);
    };
    void track(const cv::Mat& frame, const Pupil& previousPupil, Pupil& pupil,
        TrackingParameters params) override
    {
        pupil = detect(frame, params);
    }
    std::string description()
    {
        if (!pupilDetectionMethod)
            setPupilDetectionMethod(defaultPupilDetectionMethod());
        return pupilDetectionMethod->description();
    }
    std::shared_ptr<PupilDetectionMethod> defaultPupilDetectionMethod()
    {
        return std::make_shared<T>();
    }
};

#endif // TRACKINGBYDETECTION_H
