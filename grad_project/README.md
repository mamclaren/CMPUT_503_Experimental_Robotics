# CMPUT 503 - Graduate Project: Optical Flow for Robot Navigation

Group members:  Martin McLaren

[This repository contains all the code for the graduate project]

In this project, we apply optical flow analysis to the robot control problem, specifically looking at methods for estimating a mobile robot's deviation from a fixed course using the perceived motion of objects in its environment. This deviation is then used in conjunction with feedback control to correct the robot's trajectory, allowing for navigation around a simple track.

## Dependencies
The following dependencies and systems are required to run the software contained in this repository.

- Ubuntu 22.04
- ROS 2 Humble
- Duckietown Shell
- Docker
- NumPy
- OpenCV

## Code Contents

- `grad_project`: Parent directory containing all package files. Note that the project must be built from within this directory.
  - `camera_detection.py`: All code for reading and processing camera inputs and images. In particular:
    - Calculates dense optical flow field between consecutive image frames (using LK).
    - Calculates sparse optical flow field between consecutive image frames (using Farneback).
    - Draws motion vector field on current camera frame and publishes this image to a topic (view using rqt_image_view).
    - Calculates error term as average displacement, publishes to a topic. 
  - `navigation_loop.py` Primary node for executing navigation around a track.
    - Uses `camera_detection.py` to track deviation from initial trajectory.
    - Uses `odometry.py` to perform execute motion commands
    - Passes displacement error to `pid_controller.py` and adjusts motion accordingly.

## Executing

Running locally (on laptop or PC) - better for faster development/build times and more compute:
```sh
dts devel build -f

dts devel run -R csc22946 -L camera-detection -X -n "camera"
dts devel run -R csc22946 -L odometry -X -n "odometry"
dts devel run -R csc22946 -L navigation-loop -X -n "navigate"

# Or, can run all three nodes simultaneously using default launcher:
dts devel run -R csc22946 -L default -X -n "default"

```

Running directly on the Duckiebot - lower latency over the network, but less compute:
```sh
dts devel build -H csc22946 -f

dts devel run -H csc22946 -L camera-detection -n "camera"
dts devel run -H csc22946 -L odometry -n "odometry"
dts devel run -H csc22946 -L tag-loop -n "tag"

dts devel run -H csc22946 -L default -n "default
```

To kill hung docker containers:
```sh
docker rm -f <container_name>
```