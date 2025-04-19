#!/usr/bin/env python3
import numpy as np
import json
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Pose2DStamped, WheelEncoderStamped, WheelsCmdStamped, Twist2DStamped, LEDPattern
from safety_detection.srv import SetString, SetStringResponse
from std_msgs.msg import ColorRGBA, String
from Color import Color
import cv2
from cv_bridge import CvBridge
from camera_detection import CameraDetectionNode
import threading
import math

from pid_controller import simple_pid, pid_controller_v_omega

class TagLoop(DTROS):
    def __init__(self, node_name):
        super(TagLoop, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        # odometry topic
        self.ctheta = 0
        self.cpos = 0
        self.lane_error_topic = rospy.Subscriber(f"/{self.vehicle_name}/odometry", String, self.odometry_callback)

        self.outside = True

        # lane following
        self.lane_error = None
        self.lane_error_topic = rospy.Subscriber(f"/{self.vehicle_name}/lane_error", String, self.lane_error_callback)
        self.car_cmd = rospy.Publisher(f"/{self.vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.rot_dir = 1
        if self.outside:
            self.rot_dir = -1

        # tag detection
        #self.tag_list_topic = rospy.Subscriber(f"/{self.vehicle_name}/tag_list", String, self.tag_list_callback)
        self.tag_list_topic = rospy.Subscriber(f"/{self.vehicle_name}/tag_id", String, self.tag_id_callback)
        self.last_detected_tag_id = -1
        self.current_led_tag_color = -1
        stop_sign_tag_ids = [21, 22, 162, 163]
        intersection_sign_ids = [50, 15, 133, 59, 51, 56]
        ualberta_tag_ids = [94, 93, 200, 201]
        self.none_tag_id = -1
        self.tag_time_dict = {}
        for tag_id in stop_sign_tag_ids:
            self.tag_time_dict[tag_id] = 3
        for tag_id in intersection_sign_ids:
            self.tag_time_dict[tag_id] = 2
        for tag_id in ualberta_tag_ids:
            self.tag_time_dict[tag_id] = 1
        self.tag_time_dict[self.none_tag_id] = 0.5

        # ground color detection
        self.closest_red = float('inf')
        self.closest_white = float('inf')
        self.red_cooldown = 0
        self.white_cooldown = 0
        self.color_coords_topic = rospy.Subscriber(f"/{self.vehicle_name}/color_coords", String, self.color_coords_callback)

        # led commands
        self.led_command = rospy.Publisher(f"/{self.vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)
        red = ColorRGBA(r=255, g=0, b=0, a=255)
        white = ColorRGBA(r=255, g=255, b=255, a=255)
        green = ColorRGBA(r=0, g=255, b=0, a=255)
        blue = ColorRGBA(r=0, g=0, b=255, a=255)
        default_list = [white, red, white, red, white]
        all_red_list = [red] * 5
        all_green_list = [green] * 5
        all_blue_list = [blue] * 5
        all_white_list = [white] * 5
        self.all_red = LEDPattern(rgb_vals=all_red_list)
        self.all_green = LEDPattern(rgb_vals=all_green_list)
        self.all_blue = LEDPattern(rgb_vals=all_blue_list)
        self.all_white = LEDPattern(rgb_vals=all_white_list)
        self.default = LEDPattern(rgb_vals=default_list)
        self.tag_to_led = {}
        for tag_id in stop_sign_tag_ids:
            self.tag_to_led[tag_id] = self.all_red
        for tag_id in intersection_sign_ids:
            self.tag_to_led[tag_id] = self.all_blue
        for tag_id in ualberta_tag_ids:
            self.tag_to_led[tag_id] = self.all_green
        self.tag_to_led[self.none_tag_id] = self.all_white

    def lane_error_callback(self, msg):
        '''
        lane_error = {
            "lane_error": error
        }
        '''
        le_json = msg.data
        self.lane_error = json.loads(le_json)["lane_error"]
    
    def tag_list_callback(self, msg):
        '''
        [
            {
                "id": tag_id,
                "center": [x, y],
                "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "area": area
            },
            ...
        ]
        '''
        # get the tag list
        tag_list = json.loads(msg.data)
        # get the currently detected tag id with the largest area
        self.last_detected_tag_id = tag_list[0]["id"]
    
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

    def tag_id_callback(self, msg):
        '''
        msg.data = "id"
        '''
        current_tag = int(msg.data)
        if current_tag != -1:
            self.last_detected_tag_id = current_tag
    
    def color_coords_callback(self, msg):
        '''
        color_coords = {
            "red": [
                {
                'bb': [x, y, w, h],
                'center': (x, y)
                },
                ...
            ],
            "white": ...,
            "blue": ...
        }
        '''
        # get the color coords
        color_coords = json.loads(msg.data)
        # get the closest red color
        self.closest_red = min(color_coords["red"], key=lambda item: item['center'][1])['center'][1] if color_coords["red"] else float('inf')
        # get the closest white color
        self.closest_white = min(color_coords["white"], key=lambda item: item['center'][1])['center'][1] if color_coords["white"] else float('inf')
    
    def set_velocities(self, linear, rotational):
        '''
        sets the linear/rotational velocities of the Duckiebot
        linear = m/s
        rotational = radians/s
        '''
        self.car_cmd.publish(Twist2DStamped(v=linear, omega=rotational))
        rospy.loginfo(f'linear: {linear}, omega: {rotational}')

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

    def update_leds(self):
        if self.last_detected_tag_id != self.current_led_tag_color:
            self.led_command.publish(self.tag_to_led[self.last_detected_tag_id])
            rospy.loginfo(f"Changed LED from {self.current_led_tag_color} to {self.last_detected_tag_id}.")
            self.current_led_tag_color = self.last_detected_tag_id

    def tag_loop(self):
        rate_int = 10
        rate = rospy.Rate(rate_int)
        self.led_command.publish(self.all_white)
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()
            # update the leds
            self.update_leds()
            # do the lane following
            v, omega = pid_controller_v_omega(self.lane_error, simple_pid, rate_int, False)
            self.set_velocities(v, omega)
            # if the bot is at a red tape,
            if self.closest_red < 200 and self.red_cooldown == 0 and True:
                rospy.loginfo(f'detected red line, stopping. tag id: {self.last_detected_tag_id}, time to stop: {self.tag_time_dict[self.last_detected_tag_id]}')
                # update the red cooldown
                self.red_cooldown = 5
                # stop the bot
                self.pause(0.5)
                # wait for some amount of time, depending on the last seen tag id.
                rospy.sleep(self.tag_time_dict[self.last_detected_tag_id])
                # reset the last detected tag id
                self.last_detected_tag_id = self.none_tag_id
                # reset the start time, so time spent waiting is not counted
                start_time = rospy.Time.now()
                rospy.loginfo(f'done red line operations')
            # if the bot is at a white tape,
            if self.closest_white < 200 and self.white_cooldown == 0 and True:
                rospy.loginfo(f'detected white line, rotating')
                # update the white cooldown
                self.white_cooldown = 5
                # stop the bot
                self.pause(1)
                # rotate the bot
                self.rotate(math.pi/2 * 0.5, math.pi * 2)
                # stop the bot again
                self.pause(1)
                # reset the start time, so time spent waiting is not counted
                start_time = rospy.Time.now()
                rospy.loginfo(f'done white line operations')
            rate.sleep()
            # update the cooldowns
            end_time = rospy.Time.now()
            dt = (end_time - start_time).to_sec()
            rospy.loginfo(f"Loop duration: {dt:.6f} seconds")
            rospy.loginfo(f"---")
            self.red_cooldown = max(0, self.red_cooldown - dt)
            self.white_cooldown = max(0, self.white_cooldown - dt)

    def on_shutdown(self):
        # on shutdown,
        self.set_velocities(0, 0)
        self.led_command.publish(self.default)

if __name__ == '__main__':
    node = TagLoop(node_name='tag_loop')
    rospy.sleep(2)
    node.tag_loop()
    rospy.spin()
