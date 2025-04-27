# Final Exercise: Autonomous Driving in Duckietown

Group members:  Nicolas Ong, Truong Pham, Martin McLaren

For the final project, you will need to traverse the Duckietown from start to finish and collect points along the way. There are four stages of the town, each with different tasks to complete.

During the final project demo, your group will have 3 rounds to attempt the course and collect as many points as you can (as laid out in the table at the end of this document). The round with the most collected points will be counted as your final demo mark. A round is composed of 4 stages; for each stage there are potential points to collect, and the amount varies by stage. To get 100% you need to collect 100 points out of 125 potential points. Partial points will not be assigned for each subtask (i.e., you will either get 5 points for each row in the table or 0 points).

## Dependencies
The following dependencies and systems are required to run the software contained in this exercise repository.

- Ubuntu 22.04
- ROS 2 Humble
- Duckietown Shell
- Docker
- NumPy
- OpenCV
- dt_apriltags

## Code Contents

- `test_scripts` contains some test and experiment code written for this exercise.
- `packages/safety_detection/src` contains the actual ROS code that runs on the Duckiebot.
  - `camera_detection.py`: the camera detection code for this of the exercise. Expanded from previous exercise to include additional functions.
  - `odometry.py` Odometry node for movement commands.
  - `demo.py`: Primary node for running the demo. Runs nodes for each part (i.e. `part_x.py`) in consecutive order. 

## How to Run

Running locally (on laptop or PC) - better for faster development/build times and more compute. Use main launcher to start all necessary nodes:
```sh
dts devel build -f

dts devel run -R csc22946 -L default -n "default"
```

Running directly on the Duckiebot - lower latency over the network, but less compute (same
commands as above, but with -H flag):
```sh
dts devel build -H csc22946 -f

dts devel run -H csc22946 -L default-detection -n "default"
```