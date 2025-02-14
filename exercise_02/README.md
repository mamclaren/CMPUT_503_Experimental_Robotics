# Exercise 2

In this lab exercise, we apply the basic principles of kinematics and odometry to the robot control problem, performing basic movement tasks with our Duckiebot. We used the Robot Operating System (ROS) as a platform to develop, test, and run our control software, and interacted with our robot's systems using the various components of the ROS architecture. We gained a deeper understanding of the elements of the ROS system, how to use those elements to solve problems using our robot, and an appreciation of the gap between robot control concepts in principle vs. in practice.

This repository uses the ROS templates provided by Duckietown. 

## Dependencies
The following dependencies and systems are required to run the software contained in this exercise repository.

- Ubuntu 22.04
- ROS 2 Humble
- Duckietown Shell
- Docker

## Running
To execute the code provided here on your local machine (and transfer instructions to the Duckiebot over WiFi), first build the Docker container:

```
dts devel build -f
```

Then run using:

```
dts devel run -R [robot_name] -L [node_name]
```

For the movement tasks specified in this exercise, please use the node written in the ```./part_2_movement/packages/odometry/src/template_move.py``` file:  

```
dts devel run -R [robot_name] -L template_move
```