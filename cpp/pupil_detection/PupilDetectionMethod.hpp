#ifndef PUPILDETECTIONMETHOD_H
#define PUPILDETECTIONMETHOD_H

#include <bitset>
#include <deque>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../common/ocv_utils.hpp"

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

class PupilDetectionMethod {
public:
    virtual ~PupilDetectionMethod() = default;

    Pupil detect(const cv::Mat& frame, cv::Rect roi = { 0, 0, 0, 0 }, const float& userMinPupilDiameterPx = -1,
        const float& userMaxPupilDiameterPx = -1)
    {
        sanitizeROI(frame, roi);
        return implDetect(frame, roi, userMinPupilDiameterPx, userMaxPupilDiameterPx);
    }

    virtual bool hasConfidence() = 0;
    virtual bool hasCoarseLocation() = 0;
    virtual std::string description() = 0;

    // Pupil detection interface used in the tracking
    Pupil detectWithConfidence(const cv::Mat& frame, cv::Rect roi = { 0, 0, 0, 0 },
        const float& userMinPupilDiameterPx = -1, const float& userMaxPupilDiameterPx = -1);

    virtual Pupil getNextCandidate() { return Pupil(); }

    // Generic coarse pupil detection
    static cv::Rect coarsePupilDetection(const cv::Mat& frame, const float& minCoverage = 0.5f,
        const int& workingWidth = 60, const int& workingHeight = 40);

    // Generic confidence metrics
    static float outlineContrastConfidence(const cv::Mat& frame, const Pupil& pupil, const int& bias = 5);
    static float edgeRatioConfidence(const cv::Mat& edgeImage, const Pupil& pupil, std::vector<cv::Point>& edgePoints,
        const int& band = 5);
    static float angularSpreadConfidence(const std::vector<cv::Point>& points, const cv::Point2f& center);
    static float aspectRatioConfidence(const Pupil& pupil);
    static float ellipseDistanceConfidence(const Pupil& pupil, const std::vector<cv::Point>& edgePoints,
        std::vector<cv::Point>& validPoints, const int& dist = 3);

protected:
    virtual Pupil implDetect(const cv::Mat& frame, cv::Rect roi, const float& userMinPupilDiameterPx,
        const float& userMaxPupilDiameterPx)
        = 0;

    void sanitizeROI(const cv::Mat& frame, cv::Rect& roi)
    {
        cv::Rect froi = cv::Rect(0, 0, frame.cols - 1, frame.rows - 1);
        roi &= froi;
        if (roi.area() < 10) {
            roi = froi;
        }
    }
};

#endif // PUPILDETECTIONMETHOD_H
