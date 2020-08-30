#include <iostream>

#include <opencv2/highgui.hpp>

#include "PuRe.hpp"
#include "PuReST.hpp"
#include "TrackingByDetection.hpp"

using namespace std;
using namespace chrono;
using namespace cv;

static void drawOverlay(
    cv::Mat bgr, const Pupil& pupil, const float minConfidence = 0.66)
{
    if (pupil.confidence > minConfidence) {
        float g
            = 255 * (pupil.confidence - minConfidence) / (1 - minConfidence);
        float r = 255 - g;
        ellipse(bgr, pupil, Scalar(0, g, r), 2);
    }
}

static void printUsageAndExit(char* argv[])
{
    cerr << "Usage: ";
    cerr << argv[0] << " <algorithm-name> <path-to-video>";
    cerr << endl;
    exit(1);
}

int main(int argc, char* argv[])
{
    if (argc < 3) printUsageAndExit(argv);

    // Pick algorithm
    string algorithm = toLowerCase(string(argv[1]));
    unique_ptr<PupilTrackingMethod> tracker;
    if (algorithm == "pure") {
        tracker = make_unique<TrackingByDetection<PuRe>>();
    } else if (algorithm == "purest") {
        tracker = make_unique<PuReST>();
    } else {
        cerr << "Unknown algorithm: " << argv[1] << endl;
        cerr << "Expected one of [ pure, purest ]" << endl;
        printUsageAndExit(argv);
    }
    cerr << "Using: " << tracker->description() << endl;

    // Open video
    auto videoFile = string(argv[2]);
    auto cap = make_unique<VideoCapture>(videoFile);
    if (!cap->isOpened()) {
        cerr << "Could not open " << videoFile << endl;
        printUsageAndExit(argv);
    }

    // Iterate over frames
    Mat gray;
    Mat bgr;
    Pupil pupil;
    TrackingParameters params;
    Timestamp timestamp;
    cout << "x, y, width, height, angle, confidence, runtime_ms," << endl;
    while (true) {
        (*cap) >> bgr;
        if (bgr.empty()) break;

        timestamp = cap->get(CAP_PROP_POS_MSEC);

        // Prepare images
        if (bgr.channels() == 3)
            cvtColor(bgr, gray, COLOR_BGR2GRAY);
        else {
            gray = bgr;
            cvtColor(gray, bgr, COLOR_GRAY2BGR);
        }

        auto start = high_resolution_clock::now();
        tracker->detectAndTrack(timestamp, gray, pupil, params);
        auto runtime_ns
            = duration_cast<nanoseconds>(high_resolution_clock::now() - start)
                  .count();
        auto runtime_ms = static_cast<double>(1e-6 * runtime_ns);

        auto items
            = vector<double>({ pupil.center.x, pupil.center.y, pupil.size.width,
                pupil.size.height, pupil.angle, pupil.confidence, runtime_ms });
        for (const auto item : items)
            cout << item << ", ";
        cout << endl;

        drawOverlay(bgr, pupil);
        imshow("dbg", bgr);
        char c = (char)waitKey(1);
        if (c == 'q') break;
    }
}
