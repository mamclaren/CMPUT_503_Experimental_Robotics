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
from collections import deque
import math

from pid_controller import simple_pid, pid_controller_v_omega, yellow_white_pid

class TailBot(DTROS):
    def __init__(self, node_name):
        super(TailBot, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        self.cam_w, self.cam_h = 640, 480
        # lane following
        self.yellow_lane_error = None
        self.white_lane_error = None
        self.lane_error_topic = rospy.Subscriber(f"/{self.vehicle_name}/lane_error", String, self.lane_error_callback)
        self.car_cmd = rospy.Publisher(f"/{self.vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        # ground color detection
        self.closest_blue = float('inf')
        self.blue_cooldown = 0
        self.red_cooldown = 0
        self.white_cooldown = 0
        self.color_coords_topic = rospy.Subscriber(f"/{self.vehicle_name}/color_coords", String, self.color_coords_callback)

        # pedestrian detection
        self.pedestrians_detected = False
        self.pedestrians_topic = rospy.Subscriber(f"/{self.vehicle_name}/duckies_info", String, self.pedestrians_callback)

        # other bot detection
        self.other_bot_info = None
        self.other_bot_topic = rospy.Subscriber(f"/{self.vehicle_name}/other_bot_info", String, self.other_bot_callback)



        # service
        self.drive_turn_request = rospy.ServiceProxy(f'/{self.vehicle_name}/drive_turn', SetString)
        self.rotate_request = rospy.ServiceProxy(f'/{self.vehicle_name}/rotate', SetString)

        
        self.closest_red = float('inf')
        self.closest_white = float('inf')

        # tags
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

        # white line management
        self.white_line_on_oright = True
        self.state = {
            "num_red_line_we_see": 0,
            "outer_loop": False,
        }

        self.PHASE = 0

    def lane_error_callback(self, msg):
        '''
        lane_error = {
            "lane_error": error
        }
        '''
        le_json = msg.data
        self.yellow_lane_error = json.loads(le_json)["yellow_lane_error"]
        self.white_lane_error = json.loads(le_json)["white_lane_error"]
    
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
        # get the closest blue color
        self.closest_blue = min(color_coords["blue"], key=lambda item: item['center'][1])['center'][1] if color_coords["blue"] else float('inf')
        self.closest_red = min(color_coords["red"], key=lambda item: item['center'][1])['center'][1] if color_coords["red"] else float('inf')
        self.closest_white = min(color_coords["white"], key=lambda item: item['center'][1])['center'][1] if color_coords["white"] else float('inf')

        

        
    
    def pedestrians_callback(self, msg):
        '''
        pedestrians = {
            "duckie_exist": bool,
            "min_point": float
        }
        '''
        pedestrians_json = msg.data
        self.pedestrians_detected = json.loads(pedestrians_json)["duckie_exist"]
    
    def other_bot_callback(self, msg):
        '''
        other_bot = {
            other_bot_coord: , 
            bot_error: ,
            turning_left: ,
            pixel_distance
        }
        '''
        other_bot_json = msg.data
        self.other_bot_info = json.loads(other_bot_json)
        if self.other_bot_info["turning_left"]:  
            rospy.loginfo("other bot turning left")
        elif self.other_bot_info["turning_left"] == False:
            rospy.loginfo("other bot turning right")
        else:
            rospy.loginfo("other bot not turning")

    def tag_id_callback(self, msg):
        '''
        msg.data = "id"
        '''
        current_tag = int(msg.data)
        if current_tag != -1:
            self.last_detected_tag_id = current_tag
    
    def set_velocities(self, linear, rotational):
        '''
        sets the linear/rotational velocities of the Duckiebot
        linear = m/s
        rotational = radians/s
        '''
        self.car_cmd.publish(Twist2DStamped(v=linear, omega=rotational))
    
    def Tail(self):
        rate_int = 10
        rate = rospy.Rate(rate_int)
        self.lane_error_valid_before = False  
        while not rospy.is_shutdown():
            if self.yellow_lane_error is not None:   # hack
                self.lane_error_valid_before = True
            start_time = rospy.Time.now()


            if self.PHASE == 0:
                # do the lane following
                self.phase_0(rate_int)

            rate.sleep()
            # update the cooldowns
            end_time = rospy.Time.now()
            dt = (end_time - start_time).to_sec()
            self.blue_cooldown = max(0, self.blue_cooldown - dt)
            self.red_cooldown = max(0, self.red_cooldown - dt)
            self.white_cooldown = max(0, self.white_cooldown - dt)
    
    def phase_0(self, rate_int):
        # do the lane following

        if self.state["outer_loop"]:
            v, omega = self.outer_loop_pid(rate_int)
        else:
            v, omega = self.inner_loop_pid(rate_int)

        msg = f"yello error: {self.yellow_lane_error}; white error {self.white_lane_error};  v: {v} omega: {omega} "
        if self.other_bot_info is not None:
            msg += f'turn left?: {self.other_bot_info["turning_left"]} bot error {self.other_bot_info["bot_error"]}'
        #rospy.loginfo(msg)

        self.set_velocities(v, omega)

        # lane_error_valid_before is a 
        # hack for so that bot will not immediately turn right during initialization of the node. since lane_error is None at init
        #if self.yellow_lane_error is None and lane_error_valid_before: 
        #    self.drive_turn_right(math.pi / 8, -3.5, 0.2)

        # stop if the bot is too close to the other bot
        #if self.other_bot_info is not None and self.other_bot_info["pixel_distance"] <= 55:
        #    self.set_velocities(0, 0)
        
        if self.closest_red < 150 and self.red_cooldown == 0 and True:
            self.state["num_red_line_we_see"] += 1
            self.on_red_line()

            if self.state["num_red_line_we_see"] == 1:
                self.hard_code_turning()

        return
    
    def outer_loop_pid(self, rate_int):
        # follow yellow lane with white lane correction
        if self.yellow_lane_error is not None:
            v, omega = pid_controller_v_omega(self.yellow_lane_error, yellow_white_pid, rate_int, False)
        elif self.white_lane_error is not None:
            v, omega = pid_controller_v_omega(self.white_lane_error * 0.9, simple_pid, rate_int, False)
        else:
            v, omega = 0, 0
        return v, omega

    def inner_loop_pid(self, rate_int):
        # only follow yellow lane
        if self.yellow_lane_error is not None:
            v, omega = pid_controller_v_omega(self.yellow_lane_error, yellow_white_pid, rate_int, False)
        else:
            v, omega = 0, 0
            if self.lane_error_valid_before:
                rospy.loginfo("hard code turning")
                v, omega = 0.23, -0.6
        return v, omega



    def hard_code_turning(self):
        '''
        hard code the turning
        '''
        print("hard code turning")
        if self.other_bot_info is not None:
            if self.other_bot_info["turning_left"]:  
                print("other bot turning left")
                self.drive_turn_left()  # hardcoded left turn
            elif self.other_bot_info["turning_left"] == False:
                print("other bot turning right")
                self.drive_turn_right(math.pi / 2, -3.5, 0.2)  # hardcoded right turn
                return  # let bot adjust itself  using PID
            elif self.other_bot_info["turning_left"] == None:  # case: other bot isnt turning
                return  # lane follow
            
        else:
            self.drive_turn_right(math.pi / 8, -1, 0.2)
            return
    
    """
    rotate itself
    
    """
    def on_white_line(self):

        return
    
    """
    When self.closest_red < 200, stop the bot and wait for some time
    """
    def on_red_line(self):
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
        return

    def drive_turn_right(self, angle=math.pi/2, theta=-3.5, speed=0.2):
        rospy.loginfo("Turning right")
        import math
        turn_param = {
            "angle": angle,  # actual angle to turn
            "theta": theta,  # for twisted2D
            "speed": speed,
            "leds": False
        }
        self.drive_turn_request(json.dumps(turn_param))

    def drive_turn_left(self):
        rospy.loginfo("Turning left")
        import math
        turn_param = {
            "angle": math.pi / 1.8,  # actual angle to turn
            "theta": 0.5,  # for twisted2D
            "speed": 0.3,
            "leds": False
        }
        self.drive_turn_request(json.dumps(turn_param))


    def rotate(self, angle, speed):
        '''
        angle is in radians
        speed is in rad/s
        '''
        rospy.loginfo("Rotating")
        rotate_param = {
            "radians": angle,
            "speed": speed,
            "leds": False
        }
        self.rotate_request(json.dumps(rotate_param))

    
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

    def on_shutdown(self):
        # on shutdown,
        self.set_velocities(0, 0)



    def dummy(self):
        rate_int = 10
        rate = rospy.Rate(rate_int)
        rospy.loginfo("Dummy function")
        self.drive_turn_left()
        while not rospy.is_shutdown():
            rate.sleep()



if __name__ == '__main__':
    node = TailBot(node_name='tail')
    rospy.sleep(2)
    #node.drive_turn_right(math.pi / 2, -3.5, 0.2)
    #node.Tail()
    #node.drive_turn_left()
    #node.drive_turn_right()
    #node.dummy()
    rospy.spin()
