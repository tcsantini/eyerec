#ifndef OCV_UTILS_H
#define OCV_UTILS_H

/*
 * Holds useful OpenCV functions
 */

#include <opencv2/opencv.hpp>

/*
 * Macros
 */
#define CV_BLUE cv::Scalar(0xff, 0xb0, 0x00)
#define CV_GREEN cv::Scalar(0x03, 0xff, 0x76)
#define CV_RED cv::Scalar(0x00, 0x3d, 0xff)
#define CV_YELLOW cv::Scalar(0x00, 0xea, 0xff)
#define CV_CYAN cv::Scalar(0xff, 0xff, 0x18)
#define CV_MAGENT cv::Scalar(0x81, 0x40, 0xff)
#define CV_WHITE cv::Scalar(0xff, 0xff, 0xff)
#define CV_BLACK cv::Scalar(0x00, 0x00, 0x00)
#define CV_ALMOST_BLACK cv::Scalar(0x01, 0x01, 0x01)

/*
 * enums
 */
enum CVFlip { CV_FLIP_BOTH = -1,
    CV_FLIP_VERTICAL = 0,
    CV_FLIP_HORIZONTAL = 1,
    CV_FLIP_NONE = 2 };

// Rotated Rect helpers
void sincos(int angle, float& cosval, float& sinval);
std::vector<cv::Point> ellipse2Points(const cv::RotatedRect& ellipse, const int& delta = 1);
void distFromPoints(const cv::RotatedRect& ellipse, std::vector<cv::Point> points, std::vector<double>& distances);

#endif // OCV_UTILS_H
