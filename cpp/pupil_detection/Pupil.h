#ifndef ER_CPP_PUPIL_DETECTION_PUPIL_H
#define ER_CPP_PUPIL_DETECTION_PUPIL_H

#include <opencv2/core.hpp>

#include "utils.h"

class Pupil : public cv::RotatedRect {
public:
    static const int NoConfidence = -1;
    static const int SmallerThanNoConfidence = NoConfidence - 1;

    Pupil(const RotatedRect& outline, const float& confidence)
        : RotatedRect(outline)
        , confidence(confidence)
    {
    }

    Pupil(const RotatedRect& outline)
        : Pupil(outline, NoConfidence)
    {
    }

    Pupil() { clear(); }

    float confidence;

    bool hasNoConfidence() const { return confidence <= NoConfidence; }

    void clear()
    {
        angle = -1.0;
        center = { -1.0, -1.0 };
        size = { -1.0, -1.0 };
        confidence = NoConfidence;
    }

    void resize(const float& xf, const float& yf)
    {
        if (valid()) {
            center.x *= xf;
            center.y *= yf;
            size.width *= xf;
            size.height *= yf;
        }
    }
    void resize(const float& f)
    {
        if (valid()) {
            center *= f;
            size *= f;
        }
    }
    void shift(const cv::Point2f& p)
    {
        if (valid())
            center += p;
    }

    bool valid(const double& confidenceThreshold = SmallerThanNoConfidence) const
    {
        return center.x > 0 && center.y > 0 && size.width > 0 && size.height > 0 && confidence > confidenceThreshold;
    }

    bool hasOutline() const { return size.width > 0 && size.height > 0; }
    float majorAxis() const { return std::max(size.width, size.height); }
    float minorAxis() const { return std::min(size.width, size.height); }
    float axesRatio() const
    {
        auto mnmx = std::minmax<float>(size.width, size.height);
        return mnmx.first / mnmx.second;
    }
    float diameter() const { return majorAxis(); }
    float circumference() const
    {
        float a = 0.5f * majorAxis();
        float b = 0.5f * minorAxis();
        return static_cast<float>(CV_PI * abs(3.0f * (a + b) - sqrt(10.0f * a * b + 3.0f * (pow(a, 2) + pow(b, 2)))));
    }
};

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

#endif //ER_CPP_PUPIL_DETECTION_PUPIL_H
