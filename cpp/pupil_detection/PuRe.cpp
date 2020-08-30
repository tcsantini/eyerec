#include "PuRe.hpp"

#include <climits>
#include <iostream>
#include <memory>

#include <opencv2/highgui.hpp>

//#define SAVE_ILLUSTRATION

using namespace std;
using namespace cv;

PuRe::PuRe()
    : expectedFrameSize(-1, -1)
    , outlineBias(5)
    , baseSize(320, 240)
{
    /*
	 * 1) Canthi:
	 * Using measurements from white men
	 * Mean intercanthal distance 32.7 (2.4) mm
	 * Mean palpebral fissure width 27.6 (1.9) mm
	 * Jayanth Kunjur, T. Sabesan, V. Ilankovan
	 * Anthropometric analysis of eyebrows and eyelids:
	 * An inter-racial study
	 */
    meanCanthiDistanceMM = 27.6f;
    //meanCanthiDistanceMM = 32.7f;

    /*
	 * 2) Pupil:
	 * 2 to 4 mm in diameter in bright light to 4 to 8 mm in the dark
	 * Clinical Methods: The History, Physical, and Laboratory Examinations. 3rd edition.
	 * Chapter 58The Pupils
	 * Robert H. Spector.
	 */
    maxPupilDiameterMM = 8.0f;
    minPupilDiameterMM = 2.0f;

    /*
	 * 3) Iris:
	 *
	 * Extremes of 10.2 to 13 mm
	 * Caroline, P. J., and M. P. Andre.
	 * "The effect of corneal diameter on soft lens fitting"
	 * Contact Lens Spectrum 17.5 (2002): 56-56.
	 *
	 * Mean: 11.64 +/- 0.49 mm
	 * Martin, Donald K., and Brien A. Holden.
	 * "A new method for measuring the diameter of the in vivo human cornea."
	 * American journal of optometry and physiological optics 59.5 (1982): 436-441.
	 */
    maxIrisDiameterMM = 13.0f;
    meanIrisDiameterMM = 11.64f;
    minIrisDiameterMM = 10.2f;
}

PuRe::~PuRe()
{
}

void PuRe::estimateParameters(const int rows, const int cols)
{
    /*
	 * Assumptions:
	 * 1) The image contains at least both eye corners
	 * 2) The image contains a maximum of 5cm of the face (i.e., ~= 2x canthi distance)
	 */
    float d = static_cast<float>(sqrt(pow(rows, 2) + pow(cols, 2)));
    maxCanthiDistancePx = static_cast<int>(d);
    minCanthiDistancePx = static_cast<int>(2.0f * d / 3.0f);

    maxPupilDiameterPx = static_cast<int>(maxCanthiDistancePx * (maxPupilDiameterMM / meanCanthiDistanceMM));
    minPupilDiameterPx = static_cast<int>(minCanthiDistancePx * (minPupilDiameterMM / meanCanthiDistanceMM));
}

void PuRe::init(const Mat& frame)
{
    if (expectedFrameSize == Size(frame.cols, frame.rows))
        return;

    expectedFrameSize = Size(frame.cols, frame.rows);

    float rw = baseSize.width / static_cast<float>(frame.cols);
    float rh = baseSize.height / static_cast<float>(frame.rows);
    scalingRatio = min(min(rw, rh), 1.0f);
}

Mat PuRe::canny(const Mat& in, const bool blurImage, const bool useL2, const int bins, const float nonEdgePixelsRatio, const float lowHighThresholdRatio)
{
    (void)useL2;
    /*
	 * Smoothing and directional derivatives
	 * TODO: adapt sizes to image size
	 */
    if (blurImage) {
        Size blurSize(5, 5);
        GaussianBlur(in, blurred, blurSize, 1.5, 1.5, BORDER_REPLICATE);
    } else
        blurred = in;

    Sobel(blurred, dx, dx.type(), 1, 0, 7, 1, 0, BORDER_REPLICATE);
    Sobel(blurred, dy, dy.type(), 0, 1, 7, 1, 0, BORDER_REPLICATE);

    /*
	 *  Magnitude
	 */
    double minMag = 0;
    double maxMag = 0;

    cv::magnitude(dx, dy, magnitude);
    cv::minMaxLoc(magnitude, &minMag, &maxMag);

    /*
	 *  Threshold selection based on the magnitude histogram
	 */
    float low_th = 0;
    float high_th = 0;

    // Normalization
    magnitude = magnitude / maxMag;

    // Histogram
    auto histogram = std::make_unique<int[]>(bins);
    Mat res_idx = (bins - 1) * magnitude;
    res_idx.convertTo(res_idx, CV_16U);
    short* p_res_idx = nullptr;
    for (int i = 0; i < res_idx.rows; i++) {
        p_res_idx = res_idx.ptr<short>(i);
        for (int j = 0; j < res_idx.cols; j++)
            histogram[p_res_idx[j]]++;
    }

    // Ratio
    int sum = 0;
    const int nonEdgePixels = static_cast<int>(nonEdgePixelsRatio * in.rows * in.cols);
    for (int i = 0; i < bins; i++) {
        sum += histogram[i];
        if (sum > nonEdgePixels) {
            high_th = static_cast<float>(i + 1) / bins;
            break;
        }
    }
    low_th = lowHighThresholdRatio * high_th;

    /*
	 *  Non maximum supression
     */
    enum {
        NON_EDGE = 0,
        POSSIBLE_EDGE = 128,
        EDGE = 255
    };
    const float tg22_5 = 0.4142135623730950488016887242097f;
    const float tg67_5 = 2.4142135623730950488016887242097f;
    edgeType.setTo(NON_EDGE);
    for (int i = 1; i < magnitude.rows - 1; i++) {
        auto edgeTypePtr = edgeType.ptr<uchar>(i);

        const auto& p_res = magnitude.ptr<float>(i);
        const auto& p_res_t = magnitude.ptr<float>(i - 1);
        const auto& p_res_b = magnitude.ptr<float>(i + 1);
        const auto& p_x = dx.ptr<float>(i);
        const auto& p_y = dy.ptr<float>(i);

        for (int j = 1; j < magnitude.cols - 1; j++) {

            const auto& m = p_res[j];
            if (m < low_th)
                continue;

            const auto& iy = p_y[j];
            const auto& ix = p_x[j];

            const auto y = abs(iy);
            const auto x = abs(ix);
            const auto val = p_res[j] > high_th ? EDGE : POSSIBLE_EDGE;
            const auto tg22_5x = tg22_5 * x;

            if (y < tg22_5x) {
                if (m > p_res[j - 1] && m >= p_res[j + 1])
                    edgeTypePtr[j] = val;
            } else {
                float tg67_5x = tg67_5 * x;
                if (y > tg67_5x) {
                    if (m > p_res_b[j] && m >= p_res_t[j])
                        edgeTypePtr[j] = val;
                } else {
                    if ((iy <= 0) == (ix <= 0)) {
                        if (m > p_res_t[j - 1] && m >= p_res_b[j + 1])
                            edgeTypePtr[j] = val;
                    } else {
                        if (m > p_res_b[j - 1] && m >= p_res_t[j + 1])
                            edgeTypePtr[j] = val;
                    }
                }
            }
        }
    }

    /*
	 *  Hystheresis
	 */
    const int area = edgeType.cols * edgeType.rows;
    unsigned int lines_idx = 0;
    int idx = 0;

    vector<int> lines;
    edge.setTo(NON_EDGE);
    for (int i = 1; i < edgeType.rows - 1; i++) {
        for (int j = 1; j < edgeType.cols - 1; j++) {

            if (edgeType.data[idx + j] != EDGE || edge.data[idx + j] != NON_EDGE)
                continue;

            edge.data[idx + j] = EDGE;
            lines_idx = 1;
            lines.clear();
            lines.push_back(idx + j);
            unsigned int akt_idx = 0;

            while (akt_idx < lines_idx) {
                int akt_pos = lines[akt_idx];
                akt_idx++;

                if (akt_pos - edgeType.cols - 1 < 0 || akt_pos + edgeType.cols + 1 >= area)
                    continue;

                for (int k1 = -1; k1 < 2; k1++)
                    for (int k2 = -1; k2 < 2; k2++) {
                        if (edge.data[(akt_pos + (k1 * edgeType.cols)) + k2] != NON_EDGE || edgeType.data[(akt_pos + (k1 * edgeType.cols)) + k2] == NON_EDGE)
                            continue;
                        edge.data[(akt_pos + (k1 * edgeType.cols)) + k2] = EDGE;
                        lines.push_back((akt_pos + (k1 * edgeType.cols)) + k2);
                        lines_idx++;
                    }
            }
        }
        idx += edgeType.cols;
    }

    //imshow("edge", edge);
    return edge;
}

void PuRe::filterEdges(cv::Mat& edges)
{
    // TODO: there is room for improvement here; however, it is prone to small
    // mistakes; will be done when we have time
    int start_x = 5;
    int start_y = 5;
    int end_x = edges.cols - 5;
    int end_y = edges.rows - 5;

    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++) {
            uchar box[9];

            box[4] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i)]);

            if (box[4]) {
                box[1] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i)]);
                box[3] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i - 1)]);
                box[5] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i + 1)]);
                box[7] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i)]);

                if ((box[5] && box[7]))
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if ((box[5] && box[1]))
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if ((box[3] && box[7]))
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if ((box[3] && box[1]))
                    edges.data[(edges.cols * (j)) + (i)] = 0;
            }
        }

    //too many neigbours
    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++) {
            uchar neig = 0;

            for (int k1 = -1; k1 < 2; k1++)
                for (int k2 = -1; k2 < 2; k2++) {

                    if (edges.data[(edges.cols * (j + k1)) + (i + k2)] > 0)
                        neig++;
                }

            if (neig > 3)
                edges.data[(edges.cols * (j)) + (i)] = 0;
        }

    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++) {
            uchar box[17];

            box[4] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i)]);

            if (box[4]) {
                box[0] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i - 1)]);
                box[1] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i)]);
                box[2] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i + 1)]);

                box[3] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i - 1)]);
                box[5] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i + 1)]);

                box[6] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i - 1)]);
                box[7] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i)]);
                box[8] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i + 1)]);

                //external
                box[9] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i + 2)]);
                box[10] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i)]);

                box[11] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i + 3)]);
                box[12] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i + 2)]);
                box[13] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i + 2)]);

                box[14] = static_cast<uchar>(edges.data[(edges.cols * (j + 3)) + (i)]);
                box[15] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i - 1)]);
                box[16] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i + 1)]);

                if ((box[10] && !box[7]) && (box[8] || box[6])) {
                    edges.data[(edges.cols * (j + 1)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i)] = 255;
                }

                if ((box[14] && !box[7] && !box[10]) && ((box[8] || box[6]) && (box[16] || box[15]))) {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 2)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 2)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i)] = 255;
                    edges.data[(edges.cols * (j + 2)) + (i)] = 255;
                }

                if ((box[9] && !box[5]) && (box[8] || box[2])) {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j)) + (i + 1)] = 255;
                }

                if ((box[11] && !box[5] && !box[9]) && ((box[8] || box[2]) && (box[13] || box[12]))) {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i + 2)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 2)] = 0;
                    edges.data[(edges.cols * (j)) + (i + 1)] = 255;
                    edges.data[(edges.cols * (j)) + (i + 2)] = 255;
                }
            }
        }

    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++) {

            uchar box[33];

            box[4] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i)]);

            if (box[4]) {
                box[0] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i - 1)]);
                box[1] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i)]);
                box[2] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i + 1)]);

                box[3] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i - 1)]);
                box[5] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i + 1)]);

                box[6] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i - 1)]);
                box[7] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i)]);
                box[8] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i + 1)]);

                box[9] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i + 2)]);
                box[10] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i - 2)]);
                box[11] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i + 2)]);
                box[12] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i - 2)]);

                box[13] = static_cast<uchar>(edges.data[(edges.cols * (j - 2)) + (i - 1)]);
                box[14] = static_cast<uchar>(edges.data[(edges.cols * (j - 2)) + (i + 1)]);
                box[15] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i - 1)]);
                box[16] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i + 1)]);

                box[17] = static_cast<uchar>(edges.data[(edges.cols * (j - 3)) + (i - 1)]);
                box[18] = static_cast<uchar>(edges.data[(edges.cols * (j - 3)) + (i + 1)]);
                box[19] = static_cast<uchar>(edges.data[(edges.cols * (j + 3)) + (i - 1)]);
                box[20] = static_cast<uchar>(edges.data[(edges.cols * (j + 3)) + (i + 1)]);

                box[21] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i + 3)]);
                box[22] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i - 3)]);
                box[23] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i + 3)]);
                box[24] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i - 3)]);

                box[25] = static_cast<uchar>(edges.data[(edges.cols * (j - 2)) + (i - 2)]);
                box[26] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i + 2)]);
                box[27] = static_cast<uchar>(edges.data[(edges.cols * (j - 2)) + (i + 2)]);
                box[28] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i - 2)]);

                box[29] = static_cast<uchar>(edges.data[(edges.cols * (j - 3)) + (i - 3)]);
                box[30] = static_cast<uchar>(edges.data[(edges.cols * (j + 3)) + (i + 3)]);
                box[31] = static_cast<uchar>(edges.data[(edges.cols * (j - 3)) + (i + 3)]);
                box[32] = static_cast<uchar>(edges.data[(edges.cols * (j + 3)) + (i - 3)]);

                if (box[7] && box[2] && box[9])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[7] && box[0] && box[10])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[1] && box[8] && box[11])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[1] && box[6] && box[12])
                    edges.data[(edges.cols * (j)) + (i)] = 0;

                if (box[0] && box[13] && box[17] && box[8] && box[11] && box[21])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[2] && box[14] && box[18] && box[6] && box[12] && box[22])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[6] && box[15] && box[19] && box[2] && box[9] && box[23])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[8] && box[16] && box[20] && box[0] && box[10] && box[24])
                    edges.data[(edges.cols * (j)) + (i)] = 0;

                if (box[0] && box[25] && box[2] && box[27])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[0] && box[25] && box[6] && box[28])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[8] && box[26] && box[2] && box[27])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box[8] && box[26] && box[6] && box[28])
                    edges.data[(edges.cols * (j)) + (i)] = 0;

                uchar box2[18];
                box2[1] = static_cast<uchar>(edges.data[(edges.cols * (j)) + (i - 1)]);

                box2[2] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i - 2)]);
                box2[3] = static_cast<uchar>(edges.data[(edges.cols * (j - 2)) + (i - 3)]);

                box2[4] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i + 1)]);
                box2[5] = static_cast<uchar>(edges.data[(edges.cols * (j - 2)) + (i + 2)]);

                box2[6] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i - 2)]);
                box2[7] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i - 3)]);

                box2[8] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i + 1)]);
                box2[9] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i + 2)]);

                box2[10] = static_cast<uchar>(edges.data[(edges.cols * (j + 1)) + (i)]);

                box2[15] = static_cast<uchar>(edges.data[(edges.cols * (j - 1)) + (i - 1)]);
                box2[16] = static_cast<uchar>(edges.data[(edges.cols * (j - 2)) + (i - 2)]);

                box2[11] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i + 1)]);
                box2[12] = static_cast<uchar>(edges.data[(edges.cols * (j + 3)) + (i + 2)]);

                box2[13] = static_cast<uchar>(edges.data[(edges.cols * (j + 2)) + (i - 1)]);
                box2[14] = static_cast<uchar>(edges.data[(edges.cols * (j + 3)) + (i - 2)]);

                if (box2[1] && box2[2] && box2[3] && box2[4] && box2[5])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box2[1] && box2[6] && box2[7] && box2[8] && box2[9])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box2[10] && box2[11] && box2[12] && box2[4] && box2[5])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
                if (box2[10] && box2[13] && box2[14] && box2[15] && box2[16])
                    edges.data[(edges.cols * (j)) + (i)] = 0;
            }
        }
}

void PuRe::findPupilEdgeCandidates(const Mat& intensityImage, Mat& edge, vector<PupilCandidate>& candidates)
{
    /* Find all lines
	 * Small note here: using anchor points tends to result in better ellipse fitting later!
	 * It's also faster than doing connected components and collecting the labels
	 */
    vector<Vec4i> hierarchy;
    vector<vector<Point>> curves;
    findContours(edge, curves, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_KCOS);

    removeDuplicates(curves, edge.cols);

    // Create valid candidates
    for (size_t i = curves.size(); i-- > 0;) {
        PupilCandidate candidate(curves[i]);
        if (candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))
            candidates.emplace_back(candidate);
    }
}

void PuRe::combineEdgeCandidates(const cv::Mat& intensityImage, cv::Mat& edge, std::vector<PupilCandidate>& candidates)
{
    (void)edge;
    if (candidates.size() <= 1)
        return;
    vector<PupilCandidate> mergedCandidates;
    for (auto pc = candidates.begin(); pc != candidates.end(); pc++) {
        for (auto pc2 = pc + 1; pc2 != candidates.end(); pc2++) {

            Rect intersection = pc->combinationRegion & pc2->combinationRegion;
            if (intersection.area() < 1)
                continue; // no intersection
//#define DBG_EDGE_COMBINATION
#ifdef DBG_EDGE_COMBINATION
            Mat tmp;
            cvtColor(intensityImage, tmp, cv::COLOR_GRAY2BGR);
            rectangle(tmp, pc->combinationRegion, pc->color);
            for (unsigned int i = 0; i < pc->points.size(); i++)
                cv::circle(tmp, pc->points[i], 1, pc->color, -1);
            rectangle(tmp, pc2->combinationRegion, pc2->color);
            for (unsigned int i = 0; i < pc2->points.size(); i++)
                cv::circle(tmp, pc2->points[i], 1, pc2->color, -1);
            imshow("Combined edges", tmp);
            imwrite("combined.png", tmp);
            //waitKey(0);
#endif

            if (intersection.area() >= min<int>(pc->combinationRegion.area(), pc2->combinationRegion.area()))
                continue;

            vector<Point> mergedPoints = pc->points;
            mergedPoints.insert(mergedPoints.end(), pc2->points.begin(), pc2->points.end());
            PupilCandidate candidate(std::move(mergedPoints));
            if (!candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))
                continue;
            if (candidate.outlineContrast < pc->outlineContrast || candidate.outlineContrast < pc2->outlineContrast)
                continue;
            mergedCandidates.emplace_back(candidate);
        }
    }
    candidates.insert(candidates.end(), mergedCandidates.begin(), mergedCandidates.end());
}

void PuRe::searchInnerCandidates(vector<PupilCandidate>& candidates, PupilCandidate& candidate)
{
    if (candidates.size() <= 1)
        return;

    float searchRadius = 0.5f * candidate.majorAxis;
    vector<PupilCandidate> insiders;
    for (auto pc = candidates.begin(); pc != candidates.end(); pc++) {
        if (searchRadius < pc->majorAxis)
            continue;
        if (norm(candidate.outline.center - pc->outline.center) > searchRadius)
            continue;
        if (pc->outlineContrast < 0.75)
            continue;
        insiders.push_back(*pc);
    }
    if (insiders.size() <= 0) {
        //ellipse(dbg, candidate.outline, Scalar(0,255,0));
        return;
    }

    sort(insiders.begin(), insiders.end());
    candidate = insiders.back();

    //circle(dbg, searchCenter, searchRadius, Scalar(0,0,255),3);
    //candidate.draw(dbg);
    //imshow("dbg", dbg);
}

void PuRe::detect_pupil(Pupil& pupil)
{
    // 3.2 Edge Detection and Morphological Transformation
    Mat detectedEdges = canny(input, true, true, 64, 0.7f, 0.4f);

    //imshow("edges", detectedEdges);
#ifdef SAVE_ILLUSTRATION
    imwrite("edges.png", detectedEdges);
#endif
    filterEdges(detectedEdges);

    // 3.3 Segment Selection
    vector<PupilCandidate> candidates;
    findPupilEdgeCandidates(input, detectedEdges, candidates);
    if (candidates.size() <= 0)
        return;

        //for ( auto c = candidates.begin(); c != candidates.end(); c++)
        //	c->draw(dbg);

#ifdef SAVE_ILLUSTRATION
    float r = 255.0 / candidates.size();
    int i = 0;
    Mat candidatesImage;
    cvtColor(input, candidatesImage, cv::COLOR_GRAY2BGR);
    for (auto c = candidates.begin(); c != candidates.end(); c++) {
        Mat colorMat = (Mat_<uchar>(1, 1) << i * r);
        applyColorMap(colorMat, colorMat, COLORMAP_HSV);
        c->color = colorMat.at<Vec3b>(0, 0);
        c->draw(candidatesImage, c->color);
        i++;
    }
    imwrite("input.png", input);
    imwrite("filtered-edges.png", detectedEdges);
    imwrite("candidates.png", candidatesImage);
#endif

    // Combination
    combineEdgeCandidates(input, detectedEdges, candidates);
    for (auto c = candidates.begin(); c != candidates.end(); c++) {
        if (c->outlineContrast < 0.5)
            c->score = 0;
        if (c->outline.size.area() > CV_PI * pow(0.5 * maxPupilDiameterPx, 2))
            c->score = 0;
        if (c->outline.size.area() < CV_PI * pow(0.5 * minPupilDiameterPx, 2))
            c->score = 0;
    }

    /*
	for ( int i=0; i<candidates.size(); i++) {
		Mat out;
        cvtColor(input, out, cv::COLOR_GRAY2BGR);
		auto c = candidates[i];
		c.drawit(out, c.color);
		imwrite(QString("candidate-%1.png").arg(i).toStdString(), out);
		//c.drawOutlineContrast(input, 5, QString("contrast-%1-%2.png").arg(i).arg(QString::number(c.score)));
		//waitKey(0);
	}
	*/

    // Scoring
    sort(candidates.begin(), candidates.end());
    PupilCandidate selected = candidates.back();
    candidates.pop_back();

    //for ( auto c = candidates.begin(); c != candidates.end(); c++)
    //    c->draw(dbg);

    // Post processing
    searchInnerCandidates(candidates, selected);

    pupil = selected.outline;
    pupil.confidence = selected.outlineContrast;

#ifdef SAVE_ILLUSTRATION
    Mat out;
    cvtColor(input, out, cv::COLOR_GRAY2BGR);
    ellipse(out, pupil, Scalar(0, 255, 0), 2);
    line(out, Point(pupil.center.x, 0), Point(pupil.center.x, out.rows), Scalar(0, 255, 0), 2);
    line(out, Point(0, pupil.center.y), Point(out.cols, pupil.center.y), Scalar(0, 255, 0), 2);
    imwrite("out.png", out);
#endif
}

Pupil PuRe::implDetect(const cv::Mat& frame, DetectionParameters params)
{
    Pupil pupil;

    init(frame);

    estimateParameters(
        static_cast<int>(scalingRatio * frame.rows),
        static_cast<int>(scalingRatio * frame.cols));
    if (params.userMinPupilDiameterPx > 0)
        minPupilDiameterPx = static_cast<int>(scalingRatio * params.userMinPupilDiameterPx);
    if (params.userMaxPupilDiameterPx > 0)
        maxPupilDiameterPx = static_cast<int>(scalingRatio * params.userMaxPupilDiameterPx);

    // Downscaling
    Mat downscaled;
    resize(frame(params.roi), downscaled, Size(), scalingRatio, scalingRatio, cv::INTER_LINEAR);
    normalize(downscaled, input, 0, 255, NORM_MINMAX, CV_8U);

    //cvtColor(input, dbg, cv::COLOR_GRAY2BGR);

    workingSize.width = input.cols;
    workingSize.height = input.rows;

    // Preallocate stuff for edge detection
    blurred = Mat::zeros(workingSize, CV_8U);
    dx = Mat::zeros(workingSize, CV_32F);
    dy = Mat::zeros(workingSize, CV_32F);
    magnitude = Mat::zeros(workingSize, CV_32F);
    edgeType = Mat::zeros(workingSize, CV_8U);
    edge = Mat::zeros(workingSize, CV_8U);

    //cvtColor(input, dbg, cv::COLOR_GRAY2BGR);
    //circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*minPupilDiameterPx, Scalar(0,0,0), 2);
    //circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*maxPupilDiameterPx, Scalar(0,0,0), 3);

    // Detection
    detect_pupil(pupil);

    pupil.resize(1.0f / scalingRatio, 1.0f / scalingRatio);

    pupil.center += Point2f(params.roi.tl());
    //imshow("dbg", dbg);

    return pupil;
}

/*******************************************************************************
 *
 * Pupil Candidate Functions
 *
 ******************************************************************************/

inline bool PupilCandidate::isValid(const cv::Mat& intensityImage, const int& minPupilDiameterPx, const int& maxPupilDiameterPx, const int bias)
{
    if (points.size() < 5)
        return false;

    float maxGap = 0;
    for (auto p1 = points.begin(); p1 != points.end(); p1++) {
        for (auto p2 = p1 + 1; p2 != points.end(); p2++) {
            float gap = static_cast<float>(norm(*p2 - *p1));
            if (gap > maxGap)
                maxGap = gap;
        }
    }

    if (maxGap >= maxPupilDiameterPx)
        return false;
    if (maxGap <= minPupilDiameterPx)
        return false;

    outline = fitEllipse(points);
    boundaries = { 0, 0, intensityImage.cols, intensityImage.rows };

    if (!boundaries.contains(outline.center))
        return false;

    if (!fastValidityCheck(maxPupilDiameterPx))
        return false;

    pointsMinAreaRect = minAreaRect(points);
    if (ratio(pointsMinAreaRect.size.width, pointsMinAreaRect.size.height) < minCurvatureRatio)
        return false;

    if (!validityCheck(intensityImage, bias))
        return false;

    updateScore();
    return true;
}

inline bool PupilCandidate::fastValidityCheck(const int& maxPupilDiameterPx)
{
    pair<float, float> axis = minmax(outline.size.width, outline.size.height);
    minorAxis = axis.first;
    majorAxis = axis.second;
    aspectRatio = minorAxis / majorAxis;

    if (aspectRatio < minCurvatureRatio)
        return false;

    if (majorAxis > maxPupilDiameterPx)
        return false;

    combinationRegion = boundingRect(points);
    combinationRegion.width = max<int>(combinationRegion.width, combinationRegion.height);
    combinationRegion.height = combinationRegion.width;

    return true;
}

inline bool PupilCandidate::validateOutlineContrast(const Mat& intensityImage, const int& bias)
{
    outlineContrast = PupilDetectionMethod::outlineContrastConfidence(intensityImage, outline, bias);
    if (outlineContrast <= 0)
        return false;
    return true;
}

inline bool PupilCandidate::validateAnchorDistribution()
{
    anchorPointSlices.reset();
    for (auto p = points.begin(); p != points.end(); p++) {
        if (p->x - outline.center.x < 0) {
            if (p->y - outline.center.y < 0)
                anchorPointSlices.set(Q0);
            else
                anchorPointSlices.set(Q3);
        } else {
            if (p->y - outline.center.y < 0)
                anchorPointSlices.set(Q1);
            else
                anchorPointSlices.set(Q2);
        }
    }
    anchorDistribution = anchorPointSlices.count() / static_cast<float>(anchorPointSlices.size());
    return true;
}

inline bool PupilCandidate::validityCheck(const cv::Mat& intensityImage, const int& bias)
{
    mp = std::accumulate(points.begin(), points.end(), cv::Point(0, 0));
    mp.x = std::roundf(mp.x / points.size());
    mp.y = std::roundf(mp.y / points.size());

    outline.points(v);
    std::vector<cv::Point2f> pv(v, v + sizeof(v) / sizeof(v[0]));
    if (cv::pointPolygonTest(pv, mp, false) <= 0)
        return false;

    if (!validateAnchorDistribution())
        return false;

    if (!validateOutlineContrast(intensityImage, bias))
        return false;

    return true;
}
