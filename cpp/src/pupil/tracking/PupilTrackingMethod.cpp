#include "eyerec/PupilTrackingMethod.hpp"

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

Pupil PupilTrackingMethod::detectAndTrack(
    const Timestamp& ts, const cv::Mat& frame, TrackingParameters params)
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

    // Detection and tracking logic
    Pupil detectedPupil;
    Pupil trackedPupil;
    if (previousPupil.hasNoConfidence()) {

        // Detect
#ifdef EXPERIMENTAL_TRACKING
        detectedPupil = detect(frame, estimateTemporalROI(ts, params));
        // TODO: keep?
        // If detection failed, try tracking with old (e.g., during blinks)
        if (detectedPupil.confidence < minDetectionConfidence
            && !previousPupils.empty()) {
            track(frame, previousPupils.back(), trackedPupil, params);
        }
#else
        detectedPupil = detect(frame, params);
#endif

    } else {

        // Track
        track(frame, previousPupil, trackedPupil, params);

#ifdef EXPERIMENTAL_TRACKING
        // TODO: keep?
        // If tracking failed, try detection again
        if (trackedPupil.confidence < minDetectionConfidence) {
            detectedPupil = detect(frame, estimateTemporalROI(ts, params));
        }
#endif
    }

    Pupil pupil = detectedPupil.confidence > trackedPupil.confidence
        ? detectedPupil
        : trackedPupil;

    // if limits have not been set (i.e., <= 0), we don't do the size checking
    bool fitsMaxSize = params.userMaxPupilDiameterPx <= 0
        || pupil.majorAxis() <= params.userMaxPupilDiameterPx;
    bool fitsMinSize = params.userMinPupilDiameterPx <= 0
        || pupil.majorAxis() >= params.userMinPupilDiameterPx;
    if (!fitsMaxSize || !fitsMinSize) pupil.confidence = 0;

    registerPupil(ts, pupil);

    return pupil;
}

Pupil PupilTrackingMethod::detect(
    const cv::Mat& frame, DetectionParameters params)
{
    if (!pupilDetectionMethod)
        setPupilDetectionMethod(defaultPupilDetectionMethod());
    params.provideConfidence = true;
    return pupilDetectionMethod->detect(frame, params);
}

TrackingParameters PupilTrackingMethod::estimateTemporalROI(
    const Timestamp& ts, const TrackingParameters& oldParams)
{
    TrackingParameters params = oldParams;
    if (!previousPupils.empty()) {
        const auto& mostRecent = previousPupils.back();
        auto trackingRectHalfSide
            = std::max<int>(mostRecent.size.width, mostRecent.size.height);
        const auto remaining
            = 0.5 * std::max<int>(params.roi.width, params.roi.height)
            - trackingRectHalfSide;
        const auto ratio = (ts - mostRecent.ts) / static_cast<double>(maxAge);
        trackingRectHalfSide += ratio * remaining;
        Point2f delta(trackingRectHalfSide, trackingRectHalfSide);
        params.roi
            &= Rect(mostRecent.center - delta, mostRecent.center + delta);
    }
    return params;
}
