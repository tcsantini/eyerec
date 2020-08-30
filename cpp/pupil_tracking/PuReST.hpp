#ifndef PUREST_H
#define PUREST_H

#include "PuRe.hpp"
#include "PupilTrackingMethod.hpp"

class GreedyCandidate {

public:
    GreedyCandidate(const std::vector<cv::Point>& points)
        : points(points)
    {
        cv::convexHull(points, hull);
        maxGap = 0;
        meanPoint = { 0, 0 };
        for (auto p1 = hull.begin(); p1 != hull.end(); p1++) {
            meanPoint += cv::Point2f(*p1);
            for (auto p2 = p1 + 1; p2 != hull.end(); p2++) {
                float gap = norm(*p2 - *p1);
                if (gap > maxGap) maxGap = gap;
            }
        }
        meanPoint.x /= points.size();
        meanPoint.y /= points.size();
    }

    float maxGap;
    std::vector<cv::Point> points;
    std::vector<cv::Point> hull;
    cv::Point2f meanPoint;
};

class PuReST : public PupilTrackingMethod, private PuRe {

public:
    PuReST()
    {
        openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 7, 7 });
        dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 15, 15 });
    }
    static std::string desc;
    void track(const cv::Mat& frame, const Pupil& previousPupil, Pupil& pupil,
        TrackingParameters trackingParameters);
    std::string description() override
    {
        return "PuReST (santini2018pure,santini2018purest)";
    }
    std::shared_ptr<PupilDetectionMethod> defaultPupilDetectionMethod() override
    {
        return std::make_shared<PuRe>();
    }

private:
    void calculateHistogram(const cv::Mat& in, cv::Mat& histogram,
        const int& bins, const cv::Mat& mask = cv::Mat());
    void getThresholds(const cv::Mat& input, const cv::Mat& histogram,
        const Pupil& pupil, int& lowTh, int& highTh, cv::Mat& bright,
        cv::Mat& dark);
    cv::Mat dilateKernel;
    cv::Mat openKernel;
    Pupil outlineSeedPupil;

    bool greedySearch(const cv::Mat& greedyDetectorEdges,
        const Pupil& basePupil, const cv::Mat& dark, const cv::Mat& bright,
        Pupil& pupil, const float& localMinPupilDiameterPx);
    bool trackOutline(const cv::Mat& outlineTrackerEdges,
        const Pupil& basePupil, Pupil& pupil, const float& localScalingRatio,
        const float& minOutlineConfidence = 0.65f);
    void generateCombinations(const std::vector<GreedyCandidate>& seeds,
        std::vector<GreedyCandidate>& candidates, const int length);
    float confidence(const cv::Mat frame, const Pupil& pupil,
        const std::vector<cv::Point> points);
};

#endif // PUREST_H
