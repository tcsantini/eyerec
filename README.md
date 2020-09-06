# EyeRec <img src="https://raw.githubusercontent.com/tcsantini/eyerec/master/resources/icons/eyerec.svg" height=35>

Modular components for real-time mobile eye tracking.

## Table of Contents

- [Examples](#examples)
- [Available Methods](#available-methods)
- [To-be-Added Methods](#to-be-added-methods)
- [I Don't Really Want to Compile Stuff](#i-don't-really-want-to-compile-stuff)

## Examples

1. Depending on your setup, you might need to install / specify (or both) OpenCV's location. E.g.:
```bash
# Install opencv on Ubuntu:
sudo apt install libopencv-dev
# Set opencv location
export OpenCV_DIR=<opencv-path>
```

2. Compiling and running the example application for pupil tracking
```bash
mkdir build
cd build
cmake ../
make install -j 8
./cpp/examples/track-pupil <algorithm-name> <path-to-video>
```

3. The above command will also install the library, headers, and CMake config
   files, supporting cmake find\_package (e.g., see the [examples](cpp/examples/CMakeLists.txt)).
   After setting eyerec_DIR or CMAKE_PREFIX_PATH accordingly, you can use this library in your project roughly so:

```cmake
find_package(eyerec REQUIRED)
add_executable(your-app ...)
target_link_libraries(your-app eyerec::eyerec)
```

## Available Methods

### Pupil Detection

* [PuRe](https://www.sciencedirect.com/science/article/abs/pii/S1077314218300146)

### Pupil Tracking

* [PuReST](https://dl.acm.org/doi/10.1145/3204493.3204578)

## To-be-added Methods

### Calibration Methods

* (TODO) [CalibMe](https://dl.acm.org/doi/10.1145/3025453.3025950)

### Gaze Estimation Methods

* (TODO) [Grip](https://dl.acm.org/doi/abs/10.1145/3314111.3319835)

## I Don't Really Want to Compile Stuff

What you are probably looking for then is
[EyeRecToo](https://www.hci.uni-tuebingen.de/research/Projects/eyerectoo.html),
a Windows application that provides:
* real-time slippage-robust mobile eye tracking
* multiple pupil detection and tracking methods
* multiple gaze estimation methods
* high-usability gaze calibration
* recording of eye-tracking videos and data
* fiduciary marker detection
* support for multiple head-mounted eye trackers, including the popular [Pupil Core hardware](https://pupil-labs.com/products/core/).

