#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "PuRe.hpp"
#include "PuReST.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    auto videoFile = string(argv[1]);
    VideoCapture cap(videoFile);

    Pupil result;
    std::shared_ptr<PupilDetectionMethod> detector = std::make_shared<PuRe>();
    std::shared_ptr<PupilTrackingMethod> tracker = std::make_shared<PuReST>();

    Mat frame;
    Mat gray;

    double fps = 25.0;
    double timestamp = 0.0;
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        tracker->detectAndTrack(timestamp, gray, Rect(), result, detector);
        //result = detector->detect(gray, Rect());
        timestamp += fps;

        float minConfidence = 0.66;
        if (result.confidence>minConfidence) {
            float g = 255* (result.confidence - minConfidence) / (1 - minConfidence);
            float r = 255-g;
            ellipse(frame, result, Scalar(0, g, r), 2);
        }

        imshow("test", frame);
        char c = (char) waitKey(1);
        if (c=='q')
            break;
    }
    return 0;
}
