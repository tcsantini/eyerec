#include "PupilDetectionMethod.hpp"

using namespace std;
using namespace cv;

//#define DBG_COARSE_PUPIL_DETECTION
//#define DBG_OUTLINE_CONTRAST

Rect PupilDetectionMethod::coarsePupilDetection(const Mat& frame,
    const float& minCoverage, const int& workingWidth, const int& workingHeight)
{
    // We can afford to work on a very small input for haar features, but retain
    // the aspect ratio
    float xr = frame.cols / static_cast<float>(workingWidth);
    float yr = frame.rows / static_cast<float>(workingHeight);
    float r = max(xr, yr);

    Mat downscaled;
    resize(frame, downscaled, Size(), 1 / r, 1 / r, cv::INTER_LINEAR);

    auto ystep = static_cast<int>(ceil(max(0.01f * downscaled.rows, 1.0f)));
    auto xstep = static_cast<int>(ceil(max(0.01f * downscaled.cols, 1.0f)));

    auto d = static_cast<float>(
        sqrt(pow(downscaled.rows, 2) + pow(downscaled.cols, 2)));

    // Pupil radii is based on PuRe assumptions
    auto min_r = static_cast<int>((0.5 * 0.07 * d));
    auto max_r = static_cast<int>((0.5 * 0.29 * d));
    auto r_step = static_cast<int>(max(0.2f * (max_r + min_r), 1.0f));

    // TODO: padding so we consider the borders as well!

    /* Haar-like feature suggested by Swirski. For details, see
     * Świrski, Lech, Andreas Bulling, and Neil Dodgson.
     * "Robust real-time pupil tracking in highly off-axis images."
     * Proceedings of the Symposium on Eye Tracking Research and Applications.
     * ACM, 2012.
     *
     * However, we collect a per-pixel maxima instead of the global one
     */
    Mat itg;
    integral(downscaled, itg, CV_32S);
    Mat res = Mat::zeros(downscaled.rows, downscaled.cols, CV_32F);
    float best_response = std::numeric_limits<float>::min();
    deque<pair<Rect, float>> candidates;
    for (int r = min_r; r <= max_r; r += r_step) {
        int step = 3 * r;

        Point ia, ib, ic, id;
        Point oa, ob, oc, od;

        int inner_count = (2 * r) * (2 * r);
        int outer_count = (2 * step) * (2 * step) - inner_count;

        float inner_norm = 1.0f / (255 * inner_count);
        float outer_norm = 1.0f / (255 * outer_count);

        for (int y = step; y < downscaled.rows - step; y += ystep) {
            oa.y = y - step;
            ob.y = y - step;
            oc.y = y + step;
            od.y = y + step;
            ia.y = y - r;
            ib.y = y - r;
            ic.y = y + r;
            id.y = y + r;
            for (int x = step; x < downscaled.cols - step; x += xstep) {
                oa.x = x - step;
                ob.x = x + step;
                oc.x = x + step;
                od.x = x - step;
                ia.x = x - r;
                ib.x = x + r;
                ic.x = x + r;
                id.x = x - r;
                int inner = itg.ptr<int>(ic.y)[ic.x] + itg.ptr<int>(ia.y)[ia.x]
                    - itg.ptr<int>(ib.y)[ib.x] - itg.ptr<int>(id.y)[id.x];
                int outer = itg.ptr<int>(oc.y)[oc.x] + itg.ptr<int>(oa.y)[oa.x]
                    - itg.ptr<int>(ob.y)[ob.x] - itg.ptr<int>(od.y)[od.x]
                    - inner;

                float inner_mean = inner_norm * inner;
                float outer_mean = outer_norm * outer;
                float response = (outer_mean - inner_mean);

                if (response < 0.5 * best_response) continue;

                if (response < 0.5 * best_response) continue;

                if (response > best_response) best_response = response;

                if (response > res.ptr<float>(y)[x]) {
                    res.ptr<float>(y)[x] = response;
                    // The pupil is too small, the padding too large; we combine
                    // them.
                    candidates.emplace_back(make_pair(
                        Rect(0.5 * (ia + oa), 0.5 * (ic + oc)), response));
                }
            }
        }
    }

    auto compare = [](const pair<Rect, float>& a, const pair<Rect, float>& b) {
        return (a.second > b.second);
    };
    sort(candidates.begin(), candidates.end(), compare);

#ifdef DBG_COARSE_PUPIL_DETECTION
    Mat dbg;
    cvtColor(downscaled, dbg, CV_GRAY2BGR);
#endif

    // Now add until we reach the minimum coverage or run out of candidates
    Rect coarse;
    auto minWidth = static_cast<int>(minCoverage * downscaled.cols);
    auto minHeight = static_cast<int>(minCoverage * downscaled.rows);
    // for (unsigned int i = 0; i < candidates.size(); i++) {
    for (auto& c : candidates) {
        if (coarse.area() == 0)
            coarse = c.first;
        else
            coarse |= c.first;
#ifdef DBG_COARSE_PUPIL_DETECTION
        rectangle(dbg, candidates[i].first, Scalar(0, 255, 255));
#endif
        if (coarse.width > minWidth && coarse.height > minHeight) break;
    }

#ifdef DBG_COARSE_PUPIL_DETECTION
    rectangle(dbg, coarse, Scalar(0, 255, 0));
    resize(dbg, dbg, Size(), r, r);
    imshow("Coarse Detection Debug", dbg);
#endif

    // Upscale result
    coarse.x *= r;
    coarse.y *= r;
    coarse.width *= r;
    coarse.height *= r;

    // Sanity test
    Rect imRoi = Rect(0, 0, frame.cols, frame.rows);
    coarse &= imRoi;
    if (coarse.area() == 0) return imRoi;

    return coarse;
}

/* Measures the confidence for a pupil based on the inner-outer contrast
 * from the pupil following PuRe. For details, see
 * Santini, Thiago & Fuhl, Wolfgang & Kasneci, Enkelejda. (2017).
 * PuRe: Robust pupil detection for real-time pervasive eye tracking.
 * Computer Vision and Image Understanding. 10.1016/j.cviu.2018.02.002.
 */
float PupilDetectionMethod::outlineContrastConfidence(
    const Mat& frame, const Pupil& pupil, const int& bias)
{
    if (!pupil.hasOutline()) return Pupil::NoConfidence;

    Rect boundaries = { 0, 0, frame.cols, frame.rows };
    float delta = 0.15f * pupil.minorAxis();

//#define DBG_OUTLINE_CONTRAST
#ifdef DBG_OUTLINE_CONTRAST
    cv::Mat tmp;
    cv::cvtColor(frame, tmp, CV_GRAY2BGR);
    cv::ellipse(tmp, pupil, cv::Scalar(0, 255, 255));
#endif
    int evaluated = 0;
    int validCount = 0;

    vector<Point> outlinePoints = ellipse2Points(pupil, 10);
    for (auto p : outlinePoints) {
        float dx = pupil.center.x - p.x;
        float dy = pupil.center.y - p.y;

        float r = delta / sqrt(pow(dx, 2) + pow(dy, 2));
        Point2f d = { roundf(r * dx), roundf(r * dy) };
        Point2f start = Point2f(p) - d;
        Point2f end = Point2f(p) + d;

        evaluated++;
        if (!(boundaries.contains(start) && boundaries.contains(end))) continue;

        LineIterator inner(frame, start, p);
        LineIterator outer(frame, p, end);

        float innerMean = 0;
        for (int i = 0; i < inner.count; i++, ++inner)
            innerMean += *(*inner);
        innerMean /= inner.count;

        float outerMean = 0;
        for (int i = 0; i < outer.count; i++, ++outer)
            outerMean += *(*outer);
        outerMean /= outer.count;

        if (innerMean > outerMean + bias) validCount++;

#ifdef DBG_OUTLINE_CONTRAST
        if (innerMean > outerMean + bias)
            line(tmp, start, end, Scalar(0, 255, 0));
        else
            line(tmp, start, end, Scalar(0, 0, 255));
#endif
    }

    if (evaluated == 0) return 0;

#ifdef DBG_OUTLINE_CONTRAST
    cv::imshow("Outline Contrast Debug", tmp);
#endif

    return validCount / static_cast<float>(evaluated);
}

float PupilDetectionMethod::angularSpreadConfidence(
    const vector<Point>& points, const Point2f& center)
{
    enum {
        Q0 = 0,
        Q1 = 1,
        Q2 = 2,
        Q3 = 3,
    };

    std::bitset<4> anchorPointSlices;
    anchorPointSlices.reset();
    for (auto p : points) {
        if (p.x - center.x < 0) {
            if (p.y - center.y < 0)
                anchorPointSlices.set(Q0);
            else
                anchorPointSlices.set(Q3);
        } else {
            if (p.y - center.y < 0)
                anchorPointSlices.set(Q1);
            else
                anchorPointSlices.set(Q2);
        }
    }
    return anchorPointSlices.count()
        / static_cast<float>(anchorPointSlices.size());
}

float PupilDetectionMethod::aspectRatioConfidence(const Pupil& pupil)
{
    return pupil.minorAxis() / static_cast<float>(pupil.majorAxis());
}

float PupilDetectionMethod::edgeRatioConfidence(const Mat& edgeImage,
    const Pupil& pupil, vector<Point>& edgePoints, const int& band)
{
    if (!pupil.valid()) return Pupil::NoConfidence;
    Mat inBandEdges = Mat::zeros(edgeImage.rows, edgeImage.cols, CV_8U);
    ellipse(inBandEdges, pupil, Scalar(255), band);
    bitwise_and(edgeImage, inBandEdges, inBandEdges);
    findNonZero(inBandEdges, edgePoints);
    return min<float>(edgePoints.size() / pupil.circumference(), 1.0);
}

/* Measures the confidence for a pupil based on the ratio of edge points that
 * are close enough to the ellipsed fitted to those points For details, see:
 * Swirski, Lech, and Neil Dodgson.
 * "A fully-automatic, temporal approach to single camera, glint-free 3d eye
 * model fitting." Proc. PETMEI (2013).
 *
 * WARNING: not yet tested!
 */
float PupilDetectionMethod::ellipseDistanceConfidence(const Pupil& pupil,
    const std::vector<cv::Point>& edgePoints,
    std::vector<cv::Point>& validPoints, const int& dist)
{
    if (!pupil.valid() || edgePoints.empty()) return Pupil::NoConfidence;
    vector<double> distances;
    distFromPoints(pupil, edgePoints, distances);
    validPoints.clear();
    for (int i = 0; i < distances.size(); i++) {
        if (abs(distances[i]) > dist) continue;
        validPoints.push_back(edgePoints[i]);
    }
    return validPoints.size() / static_cast<float>(edgePoints.size());
}
