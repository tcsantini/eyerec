#ifndef PURE_H
#define PURE_H

#include <bitset>
#include <random>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "PupilDetectionMethod.hpp"

class PupilCandidate {
public:
    std::vector<cv::Point> points;
    cv::RotatedRect pointsMinAreaRect;
    float minCurvatureRatio;

    cv::RotatedRect outline;

    cv::Rect pointsBoundingBox;
    cv::Rect combinationRegion;
    cv::Rect br;
    cv::Rect boundaries;
    cv::Point2f v[4];
    cv::Rect outlineInscribedRect;
    cv::Point2f mp;
    float minorAxis, majorAxis;
    float aspectRatio;
    cv::Mat internalArea;
    float innerMeanIntensity;
    float outerMeanIntensity;
    float contrast;
    float outlineContrast;
    float anchorDistribution;
    float score;
    std::bitset<4> anchorPointSlices;

    cv::Scalar color;

    enum {
        Q0 = 0,
        Q1 = 1,
        Q2 = 2,
        Q3 = 3,
    };

    PupilCandidate(std::vector<cv::Point> points)
        : minCurvatureRatio(0.198912f)
        , // (1-cos(22.5))/sin(22.5)
        aspectRatio(0.0f)
        , outlineContrast(0.0f)
        , anchorDistribution(0.0f)
        , score(0.0f)
        , color(0, 255, 0)
    {
        this->points = points;
    }
    bool isValid(const cv::Mat& intensityImage, const int& minPupilDiameterPx, const int& maxPupilDiameterPx, const int bias = 5);
    void estimateOutline();
    bool isCurvatureValid();

    // Support functions
    float ratio(float a, float b)
    {
        std::pair<float, float> sorted = std::minmax(a, b);
        return sorted.first / sorted.second;
    }

    bool operator<(const PupilCandidate& c) const
    {
        return (score < c.score);
    }

    bool fastValidityCheck(const int& maxPupilDiameterPx);

    bool validateAnchorDistribution();

    bool validityCheck(const cv::Mat& intensityImage, const int& bias);

    bool validateOutlineContrast(const cv::Mat& intensityImage, const int& bias);
    bool drawOutlineContrast(const cv::Mat& intensityImage, const int& bias);

    void updateScore()
    {
        score = 0.33f * aspectRatio + 0.33f * anchorDistribution + 0.34f * outlineContrast;
        // ElSe style
        //score = (1-innerMeanIntensity)*(1+abs(outline.size.height-outline.size.width));
    }

    void draw(cv::Mat out)
    {
        //cv::ellipse(out, outline, cv::Scalar(0,255,0));
        //cv::rectangle(out, combinationRegion, cv::Scalar(0,255,255));
        for (unsigned int i = 0; i < points.size(); i++)
            cv::circle(out, points[i], 1, cv::Scalar(0, 255, 255));

        cv::circle(out, mp, 3, cv::Scalar(0, 0, 255), -1);
    }

    void draw(cv::Mat out, cv::Scalar color)
    {
        int w = 2;
        cv::circle(out, points[0], w, color, -1);
        for (unsigned int i = 1; i < points.size(); i++) {
            cv::circle(out, points[i], w, color, -1);
            cv::line(out, points[i - 1], points[i], color, w - 1);
        }
        cv::line(out, points[points.size() - 1], points[0], color, w - 1);
    }

    void drawit(cv::Mat out, cv::Scalar color)
    {
        int w = 2;
        for (unsigned int i = 0; i < points.size(); i++)
            cv::circle(out, points[i], w, color, -1);
        cv::ellipse(out, outline, color);
    }
};

class PuRe : public PupilDetectionMethod {
public:
    PuRe();
    ~PuRe();

    Pupil implDetect(const cv::Mat& frame, DetectionParameters params) override;
    bool hasConfidence() override { return true; }
    bool hasCoarseLocation() override { return false; }
    std::string description() { return "PuRe (santini2018pure)"; };

    float meanCanthiDistanceMM;
    float maxPupilDiameterMM;
    float minPupilDiameterMM;

    float maxIrisDiameterMM;
    float meanIrisDiameterMM;
    float minIrisDiameterMM;

protected:
    cv::RotatedRect detectedPupil;
    cv::Size expectedFrameSize;

    int outlineBias;

    static const cv::RotatedRect invalidPupil;

    /*
     *  Initialization
     */
    void init(const cv::Mat& frame);
    void estimateParameters(const int rows, const int cols);

    /*
     * Downscaling
     */
    cv::Size baseSize;
    cv::Size workingSize;
    float scalingRatio;

    /*
     *  Detection
     */
    void detect_pupil(Pupil& pupil);

    // Canny
    cv::Mat blurred, dx, dy, magnitude;
    cv::Mat edgeType, edge;
    cv::Mat canny(const cv::Mat& in, const bool blur = true, const bool useL2 = true, const int bins = 64, const float nonEdgePixelsRatio = 0.7f, const float lowHighThresholdRatio = 0.4f);

    // Edge filtering
    void filterEdges(cv::Mat& edges);

    // Remove duplicates (e.g., from closed loops)
    int pointHash(const cv::Point& p, const int cols) { return p.y * cols + p.x; }
    void removeDuplicates(std::vector<std::vector<cv::Point>>& curves, const int& cols)
    {
        std::map<int, uchar> contourMap;
        for (size_t i = curves.size(); i-- > 0;) {
            if (contourMap.count(pointHash(curves[i][0], cols)) > 0)
                curves.erase(curves.begin() + i);
            else {
                for (int j = 0; j < curves[i].size(); j++)
                    contourMap[pointHash(curves[i][j], cols)] = 1;
            }
        }
    }

    void findPupilEdgeCandidates(const cv::Mat& intensityImage, cv::Mat& edge, std::vector<PupilCandidate>& candidates);
    void combineEdgeCandidates(const cv::Mat& intensityImage, cv::Mat& edge, std::vector<PupilCandidate>& candidates);
    void searchInnerCandidates(std::vector<PupilCandidate>& candidates, PupilCandidate& candidate);

    cv::Mat input;
    cv::Mat dbg;

    int maxCanthiDistancePx;
    int minCanthiDistancePx;
    int maxPupilDiameterPx;
    int minPupilDiameterPx;
};

#endif // PURE_H
