#ifndef CPP_INCLUDE_EYEREC_TRACKINGBYDETECTION_H
#define CPP_INCLUDE_EYEREC_TRACKINGBYDETECTION_H

#include "eyerec/PupilTrackingMethod.hpp"

template <class T>
class TrackingByDetection : public PupilTrackingMethod {

public:
    Pupil detectAndTrack(const Timestamp& ts, const cv::Mat& frame,
        TrackingParameters params) override
    {
        return detect(frame, params);
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

#endif // CPP_INCLUDE_EYEREC_TRACKINGBYDETECTION_H
