#!/usr/bin/env python3
import numpy as np
import json
import os
import rospy

import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Pose2DStamped, WheelEncoderStamped, WheelsCmdStamped, Twist2DStamped, LEDPattern
from safety_detection.srv import SetString, SetStringResponse
from std_msgs.msg import ColorRGBA, String
from Color import Color
from cv_bridge import CvBridge

from camera_detection import CameraDetectionNode
from pid_controller import simple_pid, flow_pid, sparse_flow_pid, pid_controller_v_omega

class NavigationLoop(DTROS):
    def __init__(self, node_name):
        super(NavigationLoop, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        # Odometry topic variables
        self.ctheta = 0
        self.cpos = 0
        self.lane_error_topic = rospy.Subscriber(f"/{self.vehicle_name}/odometry", String, self.odometry_callback)

        self.outside = True

        # Lane following error
        self.lane_error = None
        self.lane_error_topic = rospy.Subscriber(f"/{self.vehicle_name}/lane_error", String, self.lane_error_callback)
        self.car_cmd = rospy.Publisher(f"/{self.vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.rot_dir = 1
        if self.outside:
            self.rot_dir = -1

        # Flow error
        self.flow_error = None
        self.flow_error_topic = rospy.Subscriber(f"/{self.vehicle_name}/flow_error", String, self.flow_error_callback)


    def lane_error_callback(self, msg):
        '''
        lane_error = {
            "lane_error": error
        }
        '''
        le_json = msg.data
        self.lane_error = json.loads(le_json)["lane_error"]


    def flow_error_callback(self, msg):
        '''
        flow_errors = {
            "flow_error": error
        }
        '''
        fe_json = msg.data
        self.flow_error = json.loads(fe_json)["flow_error"]
    

    def odometry_callback(self, msg):
        '''
        odometry_data = {
            "cpos": self.cpos,
            "ctheta": self.ctheta,
            ...
        }
        '''
        odometry_data = json.loads(msg.data)
        self.ctheta = odometry_data["ctheta"]
        self.cpos = odometry_data["cpos"]

    
    def set_velocities(self, linear, rotational):
        '''
        sets the linear/rotational velocities of the Duckiebot
        linear = m/s
        rotational = radians/s
        '''
        self.car_cmd.publish(Twist2DStamped(v=linear, omega=rotational))
        #rospy.loginfo(f'linear: {linear}, omega: {rotational}')


    def pause(self, seconds):
        '''
        seconds should be positive
        '''
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            self.set_velocities(0, 0)
            cur_time = rospy.Time.now()
            if (cur_time - start_time).to_sec() >= seconds:
                break
            rate.sleep()


    def rotate(self, radians, speed):
        '''
        radians should be positive.
        speed can be positive for clockwise,
        negative for counter-clockwise
        '''
        starting_ctheta = self.ctheta
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.set_velocities(0, speed)
            cur_radians = self.ctheta - starting_ctheta
            if cur_radians >= radians:
                break
            rate.sleep()
        self.set_velocities(0, 0)


    def navigation_loop(self):
        rate_int = 10
        rate = rospy.Rate(rate_int)
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Perform traditional lane following
            #v, omega = pid_controller_v_omega(self.lane_error, simple_pid, rate_int, False)
            #self.set_velocities(v, omega)

            # Drive in a straight line with optical flow
            v, omega = pid_controller_v_omega(self.flow_error, flow_pid, rate_int, False)
            self.set_velocities(v, omega)

            rate.sleep()

            end_time = rospy.Time.now()
            dt = (end_time - start_time).to_sec()
            #rospy.loginfo(f"Loop duration: {dt:.6f} seconds")
            #rospy.loginfo(f"---")


    def on_shutdown(self):
        # on shutdown,
        self.set_velocities(0, 0)
        #self.led_command.publish(self.default)

if __name__ == '__main__':
    node = NavigationLoop(node_name='navigation_loop')
    rospy.sleep(2)
    node.navigation_loop()
    rospy.spin()
