#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun optical_flow_navigation navigation_loop.py

# wait for app to end
dt-launchfile-join
