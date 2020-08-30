#include "PupilTrackingMethod.hpp"

using namespace std;
using namespace cv;

#define EXPERIMENTAL_TRACKING

void PupilTrackingMethod::reset()
{
    previousPupils.clear();
    previousPupil = TrackedPupil();
}

void PupilTrackingMethod::registerPupil(const Timestamp& ts, Pupil& pupil)
{
    if (pupil.confidence > minDetectionConfidence) {
        previousPupil = TrackedPupil(ts, pupil);
        previousPupils.emplace_back(previousPupil);
    } else
        previousPupil = TrackedPupil();
}

void PupilTrackingMethod::detectAndTrack(const Timestamp& ts, const cv::Mat& frame, const cv::Rect& roi, Pupil& pupil, const float& minPupilDiameterPx, const float& maxPupilDiameterPx)
{
    cv::Size frameSize = { frame.cols, frame.rows };
    if (expectedFrameSize != frameSize) {
        // Reference frame changed. Let's start over!
        expectedFrameSize = frameSize;
        reset();
    }

    // Remove old samples
    while (!previousPupils.empty()) {
        if (ts - previousPupils.front().ts > maxAge)
            previousPupils.pop_front();
        else
            break;
    }

    pupil.clear();

    // Detection and tracking logic
    Pupil alternative;
    if (previousPupil.hasNoConfidence()) {

        // Detect

#ifdef EXPERIMENTAL_TRACKING
        pupil = detect(frame, estimateTemporalROI(ts, roi), minPupilDiameterPx, maxPupilDiameterPx);
        // TODO: keep?
        // If detection failed, try tracking with old (e.g., during blinks)
        if (pupil.confidence < minDetectionConfidence && !previousPupils.empty()) {
            track(frame, roi, previousPupils.back(), alternative, minPupilDiameterPx, maxPupilDiameterPx);
            if (alternative.confidence > pupil.confidence)
                pupil = alternative;
        }
#else
        pupil = detect(frame, roi, minPupilDiameterPx, maxPupilDiameterPx);
#endif

    } else {

        // Track
        track(frame, roi, previousPupil, pupil, minPupilDiameterPx, maxPupilDiameterPx);

#ifdef EXPERIMENTAL_TRACKING
        // TODO: keep?
        // If tracking failed, try detection again
        if (pupil.confidence < minDetectionConfidence) {
            alternative = detect(frame, estimateTemporalROI(ts, roi), minPupilDiameterPx, maxPupilDiameterPx);
            if (alternative.confidence > pupil.confidence)
                pupil = alternative;
        }
#endif
    }

    // if limits have not been set (i.e., <= 0), we don't do the size checking
    bool fitsMaxSize = maxPupilDiameterPx <= 0 || pupil.majorAxis() <= maxPupilDiameterPx;
    bool fitsMinSize = minPupilDiameterPx <= 0 || pupil.majorAxis() >= minPupilDiameterPx;
    if (!fitsMaxSize || !fitsMinSize)
        pupil.confidence = 0;

    registerPupil(ts, pupil);
}

Pupil PupilTrackingMethod::detect(const cv::Mat& frame, const cv::Rect& roi, const float& minPupilDiameterPx, const float& maxPupilDiameterPx)
{
    if (!pupilDetectionMethod)
        setPupilDetectionMethod(defaultPupilDetectionMethod());
    return pupilDetectionMethod->detectWithConfidence(frame, roi, minPupilDiameterPx, maxPupilDiameterPx);
}

cv::Rect PupilTrackingMethod::estimateTemporalROI(const Timestamp& ts, const cv::Rect& roi)
{
    Rect trackingRect = roi;
    if (!previousPupils.empty()) {
        const auto& mostRecent = previousPupils.back();
        auto trackingRectHalfSide = std::max<int>(mostRecent.size.width, mostRecent.size.height);
        const auto remaining = 0.5 * std::max<int>(roi.width, roi.height) - trackingRectHalfSide;
        const auto ratio = (ts - mostRecent.ts) / static_cast<double>(maxAge);
        trackingRectHalfSide += ratio * remaining;
        Point2f delta(trackingRectHalfSide, trackingRectHalfSide);
        trackingRect &= Rect(mostRecent.center - delta, mostRecent.center + delta);
    }
    return trackingRect;
}
