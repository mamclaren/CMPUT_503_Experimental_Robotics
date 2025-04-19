#!/usr/bin/env python3
import json
import math
import numpy as np
import os
import time

import cv2
from cv_bridge import CvBridge
from Color import Color

import rospy
from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import CompressedImage, Image
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Pose2DStamped, WheelEncoderStamped, WheelsCmdStamped, Twist2DStamped, LEDPattern

class CameraDetectionNode(DTROS):
    def __init__(self, node_name):
        super(CameraDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        # camera subscriber
        self.camera_image = None
        self.bridge = CvBridge()
        self.camera_sub = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.camera_callback)

        # Publishers
        self.color_coords_topic = rospy.Publisher(f"/{self.vehicle_name}/color_coords", String, queue_size=1)
        self.lane_error_topic = rospy.Publisher(f"/{self.vehicle_name}/lane_error", String, queue_size=1)
        self.camera_detection_image_topic = rospy.Publisher(f"/{self.vehicle_name}/camera_detection_image", Image, queue_size=1)



        # OpticalFlow global variables ---------------------------------------------------------------

        self.prev_frame = None
        self.optical_flow_loop_count = 0

        self.line_array = []
        self.cumulative_u_displacement = 0.0
        self.cumulative_v_displacement = 0.0

        self.lk_parameters = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.lk_features = None
        self.lk_loss_threshold = 0.3

        self.flow_image = rospy.Publisher(f"/{self.vehicle_name}/flow_image", Image, queue_size=1)
        self.flow_error = rospy.Publisher(f"/{self.vehicle_name}/flow_error", String, queue_size=1)

        #----------------------------------------------------------------------------------------------


        # Camera matrix and distortion coefficients from intrinsic.yaml file
        self.cam_matrix = np.array([
            [319.2461317458548, 0.0, 307.91668484581703],
            [0.0, 317.75077109798957, 255.6638447529814],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeff = np.array([-0.25706255601943445, 0.045805679651939275, -0.0003584336283982042, -0.0005756902051068707, 0.0])
        
        # From extrinsic.yaml file
        self.homography = np.array([
            [-0.00013668875104344582, 0.0005924050290243054, -0.5993724660928124],
            [-0.0022949507610645035, -1.5331615246117395e-05, 0.726763100835842],
            [0.00027302496335237673, 0.017296161892938217, -2.946528752705874]
        ])

        # Projection to ground plane homography matrix
        self.cam_w, self.cam_h = 640, 480
        self.ground_w, self.ground_h = 1250, 1250
        dst_pts_translation = np.array([(self.ground_w / 2) - 24, self.ground_h - 255], dtype=np.float32)
        src_pts = np.array([[284, 285], [443, 285], [273, 380], [584, 380]], dtype=np.float32)
        dst_pts = np.array([[0, 0], [186, 0], [0, 186], [186, 186]], dtype=np.float32)
        dst_pts = dst_pts + dst_pts_translation
        self.homography_to_ground, _ = cv2.findHomography(src_pts, dst_pts)

        # Robot position in the projected ground plane,
        # Below the center of the image by some distance (mm)
        self.robot_x, self.robot_y = self.ground_w / 2, self.ground_h + 100

        # Color Detection Stuff
        # color detection parameters in HSV format
        # Define range
        hr = 40
        sr = 50
        vr = 50
        #red_hsv = np.array([0, 144, 255])  # rGB: 255, 111, 111
        #self.red_lower = np.array([max(0, red_hsv[0] - hr), max(0, red_hsv[1] - sr), max(0, red_hsv[2] - vr)])
        #self.red_upper = np.array([min(179, red_hsv[0] + hr), min(255, red_hsv[1] + sr), min(255, red_hsv[2] + vr)])
        self.red_lower = np.array([136, 87, 111], np.uint8)
        self.red_upper = np.array([180, 255, 255], np.uint8)

        self.green_lower = np.array([34, 52, 72], np.uint8)
        self.green_upper = np.array([82, 255, 255], np.uint8)

        self.blue_lower = np.array([110, 80, 120], np.uint8)
        self.blue_upper = np.array([130, 255, 255], np.uint8)

        self.yellow_lower = np.array([21, 100, 60*2.55], np.uint8)
        self.yellow_higher = np.array([33, 255, 100*2.55], np.uint8)

        self.white_lower = np.array([0, 0, 180], np.uint8)
        self.white_higher = np.array([180, 50, 255], np.uint8)

        self.orange_lower = np.array([30/2, 50*2.55, 30*2.55], np.uint8)
        self.orange_higher = np.array([36/2, 100*2.55, 100*2.55], np.uint8)
        # color bounds
        self.color_bounds = {
            Color.RED: (self.red_lower, self.red_upper),
            Color.BLUE: (self.blue_lower, self.blue_upper),
            Color.GREEN: (self.green_lower, self.green_upper),
            Color.YELLOW: (self.yellow_lower, self.yellow_higher),
            Color.WHITE: (self.white_lower, self.white_higher),
            Color.ORANGE: (self.orange_lower, self.orange_higher),
        }

        # color to BGR dictionary
        self.color_to_bgr = {
            Color.RED : (0, 0, 255),
            Color.BLUE: (255, 0, 0),
            Color.GREEN: (0, 255, 0),
            Color.WHITE: (255, 255, 255),
            Color.YELLOW: (0, 255, 255),
            Color.BLACK: (0, 0, 0),
            Color.ORANGE: (0, 165, 255)
        }
        
        # Draw Toggles
        self.draw_lane_detection = True
        self.draw_bounding_boxes = True

        # if the bot puts the white line on the right or left
        self.white_on_right = True

        # offset for simple lane detection
        self.simple_offset = 100

        # threshold for ground/horizon
        self.horizon = self.cam_h * 0.6

        # masks for ground bounding box detection
        self.polygon_points = np.array([
            [60 * self.cam_w // 100, self.cam_h/2],   # Top-right
            [40 * self.cam_w // 100, self.cam_h/2],   # Top-left
            [5 * self.cam_w // 100, self.cam_h],      # Bottom-left
            [95 * self.cam_w // 100, self.cam_h],     # Bottom-right
        ], np.int32)
        self.lane_mask = np.zeros((self.cam_h, self.cam_w), dtype=np.uint8)
        cv2.fillPoly(self.lane_mask, [self.polygon_points], 255)

        self.polygon_points_white = np.array([
            [55 * self.cam_w // 100, self.cam_h/2],   # Top-right
            [45 * self.cam_w // 100, self.cam_h/2],   # Top-left
            [20 * self.cam_w // 100, self.cam_h],     # Bottom-left
            [80 * self.cam_w // 100, self.cam_h],     # Bottom-right
        ], np.int32)
        self.lane_mask_white = np.zeros((self.cam_h, self.cam_w), dtype=np.uint8)
        cv2.fillPoly(self.lane_mask_white, [self.polygon_points_white], 255)

    def id_callback(self, msg):
        self.tag_id = int(msg.data)

    def camera_callback(self, msg):
        # convert compressed image to cv2
        cv2_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        # save the raw camera image
        self.camera_image = cv2_image
    
    def undistort_image(self, cv2_img):
        h, w = cv2_img.shape[:2]
        # optimal camera matrix lets us see entire camera image (image edges cropped without), but some distortion visible
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix, self.dist_coeff, (w,h), 0, (w,h))

        # undistorted image using calibration parameters
        return cv2.undistort(cv2_img, self.cam_matrix, self.dist_coeff, None)
    
    def get_color_mask(self, color: Color, cv2_img):
        '''
        the color mask gets all the pixels in the image that are within the color bounds
        the color mask is an ndarray of shape (h, w) with values 0 or 255.
        0 means the pixel is not in the color bounds, 255 means it is
        '''
        hsv_frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        # Get the lower and upper bounds for the given color
        lower, upper = self.color_bounds.get(color, (None, None))
        assert lower is not None and upper is not None, f"Invalid color: {color}"

        # Create color mask
        color_mask = cv2.inRange(hsv_frame, lower, upper)
        color_mask = cv2.dilate(color_mask, kernel)

        return color_mask
    
    def get_contours(self, color_mask):
        '''
        using the color mask, we can get the contours of the color
        the contours are the edges of the color, defined by a list of points
        contours is a tuple of ndarrays of shape (n, 1, 2)
        '''
        contours, hierarchy = cv2.findContours(color_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    
    def get_ground_bounding_boxes(self, color, cv2_img):
        # get the color mask
        color_mask = self.get_color_mask(color, cv2_img)
        # only look at colors in the lane:
        color_mask = cv2.bitwise_and(color_mask, self.lane_mask)
        if color == Color.WHITE:
            color_mask = cv2.bitwise_and(color_mask, self.lane_mask_white)
        # get the color contours
        contours, hierarchy = self.get_contours(color_mask)
        # get all the bounding boxes with a large area, and on the ground.
        bbs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if (area > 300): 
                x, y, w, h = cv2.boundingRect(contour)
                contour_bb = (x, y, w, h)
                contour_center = (x + w / 2, y + h / 2)
                if contour_center[1] > self.horizon:
                    bbs.append({"bb": contour_bb, "center": contour_center})
        return bbs
    
    def project_point_to_ground(self, point):
        '''
        point is a tuple of (x, y) coordinates
        the point is relative to the bot.
        '''
        point = np.array([point], dtype=np.float32)
        new_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.homography_to_ground)
        new_point = new_point.ravel()
        new_point = (new_point[0] - self.robot_x, -(new_point[1] - self.robot_y))
        return new_point
    
    def project_bounding_box_to_ground(self, bounding_box):
        '''
        bounding_box is a tuple of (x, y, w, h) coordinates
        output is a list of 4 points in the ground plane.
        These points are relative to the robot.
        '''
        if not bounding_box: return None
        x, y, w, h = bounding_box
        points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        new_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.homography_to_ground)
        new_points = new_points.reshape(-1, 2)
        transformed_coords = np.column_stack([
            new_points[:, 0] - self.robot_x,
            -(new_points[:, 1] - self.robot_y)
        ])
        return transformed_coords
    
    def project_bounding_boxes_to_ground(self, bbs):
        pbbs = []
        for bb in bbs:
            pbb = self.project_bounding_box_to_ground(bb["bb"])
            pc = self.project_point_to_ground(bb["center"])
            data = {"bb": pbb, "center": pc}
            serializable_data = {
                'bb': data['bb'].tolist(),  # Convert the np.array to a Python list
                'center': (float(data['center'][0]), float(data['center'][1]))  # Convert np.float32 to float
            }
            pbbs.append(serializable_data)
        return pbbs
    
    def draw_bounding_boxes_to_image(self, image, bbs, projected_bbs, color):
        for i in range(len(bbs)):
            bb = bbs[i]
            pbb = projected_bbs[i]
            self.draw_bounding_box(image, bb["bb"], bb["center"], pbb["center"], color)

    def perform_ground_color_detection(self, clean_image, draw_image):
        if self.camera_image is None: return draw_image
        # get all the bounding boxes above a certain area for each color (white, red, blue)
        # also in the bottom half of the image
        red_bbs = self.get_ground_bounding_boxes(Color.RED, clean_image)
        white_bbs = self.get_ground_bounding_boxes(Color.WHITE, clean_image)
        blue_bbs = self.get_ground_bounding_boxes(Color.BLUE, clean_image)
        # project the bounding boxes (and their centers) to the ground
        red_bbs_p = self.project_bounding_boxes_to_ground(red_bbs)
        white_bbs_p = self.project_bounding_boxes_to_ground(white_bbs)
        blue_bbs_p = self.project_bounding_boxes_to_ground(blue_bbs)
        # publish the info to the topic
        color_coords = {
            "red": red_bbs_p,
            "white": white_bbs_p,
            "blue": blue_bbs_p
        }
        json_coords = json.dumps(color_coords)
        self.color_coords_topic.publish(json_coords)
        # draw the bounding boxes and their centers, with their center ground coordinates
        if self.draw_bounding_boxes:
            cv2.line(draw_image, tuple(self.polygon_points[0]), tuple(self.polygon_points[3]), (0, 255, 0), 2)  # Left line
            cv2.line(draw_image, tuple(self.polygon_points[1]), tuple(self.polygon_points[2]), (0, 255, 0), 2)  # Right line
            cv2.line(draw_image, tuple(self.polygon_points_white[0]), tuple(self.polygon_points_white[3]), (0, 255, 0), 2)  # Left line
            cv2.line(draw_image, tuple(self.polygon_points_white[1]), tuple(self.polygon_points_white[2]), (0, 255, 0), 2)  # Right line
            self.draw_bounding_boxes_to_image(draw_image, red_bbs, red_bbs_p, Color.RED)
            self.draw_bounding_boxes_to_image(draw_image, white_bbs, white_bbs_p, Color.WHITE)
            self.draw_bounding_boxes_to_image(draw_image, blue_bbs, blue_bbs_p, Color.BLUE)
        return draw_image
    
    def get_largest_bounding_box(self, color, cv2_img):
        # get the color mask
        color_mask = self.get_color_mask(color, cv2_img)
        # get the color contours
        contours, hierarchy = self.get_contours(color_mask)
        # get the largest bounding box
        largest_bb = None
        largest_area = -float('inf')
        for contour in contours:
            area = cv2.contourArea(contour)
            if (area > largest_area):
                x, y, w, h = cv2.boundingRect(contour)
                largest_area = area
                largest_bb = (x, y, w, h)
        return largest_bb
    
    def draw_vertical_line(self, image, x, color):
        '''
        draws a vertical line at the given x-coordinate
        '''
        x = int(x)
        cv2.line(image, (x, 0), (x, image.shape[0]), color=self.color_to_bgr[color], thickness=1)
    
    def draw_bounding_box(self, image, bb, center, coords, color):
        '''
        this function draws the bounding box and the ground x, y coordinates
        '''
        if bb is None: return
        # draw the bounding box
        x, y, w, h = bb
        cv2.rectangle(image, (x, y), (x + w, y + h), self.color_to_bgr[color], 2) 
        # draw the center
        cv2.circle(image, (int(center[0]), int(center[1])), radius=2, color=self.color_to_bgr[color], thickness=-1)
        # draw the x, y coordinates
        cv2.putText(image, f"({coords[0]:.2f}, {coords[1]:.2f})", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_to_bgr[color])
    
    def draw_lane_error_value(self, image, lane_error):
        '''
        this function draws the lane error values on the image
        '''
        if lane_error is not None: lane_error = round(lane_error, 2)
        cv2.putText(image, f"Lane Error: {lane_error}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_to_bgr[Color.BLUE])
    
    def perform_simple_lane_detection(self, clean_image, draw_image):
        if self.camera_image is None: return draw_image
        # crop image to a strip around the bottom
        image = clean_image[int(self.cam_h * 0.7):int(self.cam_h * 0.9), int(0):int(self.cam_w)]
        # crop the left or right half off
        x_offset = 0
        if self.white_on_right:
            x_offset = int(self.cam_w / 2)
            image = image[:, int(self.cam_w / 2):int(self.cam_w)]
        else:
            image = image[:, int(0):int(self.cam_w / 2)]
        # do color detection for the white line, get the biggest white blob
        white_bb = self.get_largest_bounding_box(Color.WHITE, image)
        white_center = None
        if white_bb is not None:
            white_center = (white_bb[0] + white_bb[2] / 2, white_bb[1] + white_bb[3] / 2)
        # get its distance from the left side of the image, plus some offset
        error = None
        if white_center is not None:
            if self.white_on_right:
                # negative error - bot should turn left.
                error = white_center[0] - (self.cam_w / 2 - self.simple_offset)
            else:
                error = white_center[0] - (0 + self.simple_offset)
        # publish this as an error in the lane errors topic
        lane_errors = {
            "lane_error": error
        }
        json_le = json.dumps(lane_errors)
        self.lane_error_topic.publish(json_le)
        # draw image for visualization
        if self.draw_lane_detection:
            line_1 = int(self.simple_offset + x_offset)
            cv2.line(draw_image, (line_1, int(self.cam_h * 0.7)), (line_1, int(self.cam_h * 0.9)), color=self.color_to_bgr[Color.BLUE], thickness=1)
            line_2 = int(self.cam_w / 2 - self.simple_offset + x_offset)
            cv2.line(draw_image, (line_2, int(self.cam_h * 0.7)), (line_2, int(self.cam_h * 0.9)), color=self.color_to_bgr[Color.BLUE], thickness=1)
            if white_bb is not None:
                draw_white_bb = list(white_bb)
                draw_white_bb[0] += int(x_offset)
                draw_white_bb[1] += int(self.cam_h * 0.7)
                draw_white_center = list(white_center)
                draw_white_center[0] += int(x_offset)
                draw_white_center[1] += int(self.cam_h * 0.7)
                self.draw_bounding_box(draw_image, draw_white_bb, draw_white_center, white_center, Color.BLUE)
            self.draw_lane_error_value(draw_image, error)
        return draw_image
    

    #------------------------------------------------------------------------------------------------------

    def draw_flow(self, img, flow, grid_step=16, mode='grid', imgHasColor=False):
        
        if imgHasColor:
            flow_img = img
        else:
            flow_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if mode == 'grid':
            frame_height, frame_width = img.shape[:2]
            y, x = np.mgrid[grid_step/2:frame_height:grid_step, grid_step/2:frame_width:grid_step].reshape(2,-1).astype(int)
            x_displaced, y_displaced = flow[y,x].T

            lines = np.vstack([x, y, x-x_displaced, y-y_displaced]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)

            cv2.polylines(flow_img, lines, 0, (0, 255, 0))

            for (x1, y1), (_x2, _y2) in lines:
                #arrows.append([int(x1), int(y1), float(np.sqrt((_x2-x1)*(_x2-x1) + (_y2-y1)*(_y2-y1)))])
                cv2.circle(flow_img, (x1, y1), 1, (0, 255, 0), -1)

            self.line_array = lines

        elif mode == 'hsv':
            frame_height, frame_width = flow.shape[:2]
            x_displaced, y_displaced = flow[:,:,0], flow[:,:,1]

            # Convert Cartesian flow vectors to polar coordinate form
            angle = np.arctan2(y_displaced, x_displaced) + np.pi
            magnitude = np.sqrt(x_displaced*x_displaced + y_displaced*y_displaced)

            hsv = np.zeros((frame_height, frame_width, 3), np.uint8)
            hsv[...,0] = angle * (180/np.pi/2)
            hsv[...,1] = 255
            # Normalize magnitudes to 0-255 HSV values
            hsv[...,2] = np.minimum(magnitude*4, 255)
            flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_img

    
    def perform_optical_flow_detection(self, clean_image, draw_image):
        if self.camera_image is None: 
            return clean_image
        
        # Image preprocessing
        # Smooth using Gaussian blurring
        curr_frame = cv2.GaussianBlur(clean_image, (5,5), 0)
        # Crop out top-half of frame
        #curr_frame = curr_frame[int(self.cam_h * 0.5):int(self.cam_h * 1.0), int(0):int(self.cam_w)]
        curr_frame = curr_frame[int(self.cam_h * 0.5):int(self.cam_h * 1.0), int(self.cam_w * 0.15):int(self.cam_w*0.85)]
        # Convert to grayscale
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # Reduce image size (significantly seeds up processing for dense optical flow)
        curr_frame = cv2.resize(curr_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        #clean_image = cv2.resize(clean_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
        
        curr_height, curr_width = curr_frame.shape[:2]

        if self.optical_flow_loop_count == 0:
            self.prev_frame = curr_frame
            self.flow_image.publish(self.bridge.cv2_to_imgmsg(draw_image, encoding="bgr8"))
            self.optical_flow_loop_count += 1
            return draw_image

        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, curr_frame, None, 0.5, 3, 50, 3, 5, 1.1, 0)

        draw_image = self.draw_flow(curr_frame, flow)
        #draw_image = self.draw_flow(clean_image, flow, imgHasColor=True)

        # Calculate mean displacement
        l = np.array(self.line_array, dtype='float32')
        l = l.mean(axis=0)
        l = l.flatten()
        l_start = (int(l[0]), int(l[1]))
        l_end = (math.ceil(l[2]), math.ceil(l[3]))

        # Mean displacement along x-axis
        u = l[0] - l[2]
        # Mean displacement along y-axis
        v = l[1] - l[3]

        # Track accumulated displacement
        self.cumulative_u_displacement += u*0.5
        self.cumulative_v_displacement += v
        
        str_err_u = f'Cumulative x displacement: {self.cumulative_u_displacement:.2f}'
        str_err_v = f'Cumulative y displacement: {self.cumulative_v_displacement:.2f}'

        print(str_err_u)

        # Draw mean flow and print errors onto image
        draw_image = cv2.arrowedLine(draw_image, l_end, l_start, (0, 0, 255), 2)        
        draw_image = cv2.putText(draw_image, str_err_u, (10, curr_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (235,0,255), 1)
        draw_image = cv2.putText(draw_image, str_err_v, (10, curr_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (235,0,255), 1)

        # Publish data to topics
        self.flow_image.publish(self.bridge.cv2_to_imgmsg(draw_image, encoding="bgr8"))
        flow_errors = {
            "flow_error": self.cumulative_u_displacement
        }
        json_fe = json.dumps(flow_errors)
        self.flow_error.publish(json_fe)

        # Save current frame for next iteration
        self.optical_flow_loop_count += 1
        self.prev_frame = curr_frame

        return draw_image
    
    
    def perform_sparse_optical_flow_detection(self, clean_image, draw_image):
        if self.camera_image is None:
            return clean_image
        
        # Image preprocessing
        # Smooth using Gaussian blurring
        curr_frame = cv2.GaussianBlur(clean_image, (5,5), 0)
        # Crop out top-half of frame *** COMMENT OUT FOR COLOR IMAGE DRAWING
        #curr_frame = curr_frame[int(self.cam_h * 0.5):int(self.cam_h * 1.0), int(0):int(self.cam_w)]
        # Convert to grayscale *** COMMENT OUT FOR COLOR IMAGE DRAWING
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # Reduce image size (significantly seeds up processing for dense optical flow)
        #curr_frame = cv2.resize(curr_frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        
        if self.optical_flow_loop_count == 0:
            self.lk_features = cv2.goodFeaturesToTrack(curr_frame, mask=None, **{
                'maxCorners' : 25,
                'qualityLevel' : 0.1,
                'minDistance' : 5,
                'blockSize' : 5
            })
            self.lk_features = np.float32(self.lk_features)
            self.optical_flow_loop_count += 1
            self.prev_frame = curr_frame
            self.flow_image.publish(self.bridge.cv2_to_imgmsg(draw_image, encoding="bgr8"))
            return draw_image
        
        flow, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, curr_frame, self.lk_features, None, **self.lk_parameters)

        displaced_points = flow[status == 1]
        prev_features_pos = self.lk_features[status == 1]

        # Check if features of interest have been lost, redetect as necessary
        num_lost_features = len(self.lk_features) - len(displaced_points)
        lost_feature_ratio = num_lost_features / len(self.lk_features) if len(self.lk_features) > 0 else 0

        if lost_feature_ratio > self.lk_loss_threshold:
            self.optical_flow_loop_count = 0
            return draw_image

        # Draw flow vectors
        for i, (new, old) in enumerate(zip(displaced_points, prev_features_pos)):
            y_displaced, x_displaced = new.ravel()
            y, x = old.ravel()
            draw_image = cv2.arrowedLine(draw_image, (int(y), int(x)), (int(y_displaced), int(x_displaced)), (0,255,0), 2)

        # Compute average flow vector
        curr_height, curr_width = curr_frame.shape[:2]
        curr_center = (curr_width//2, curr_height//2)
        l = np.mean(displaced_points - prev_features_pos, axis=0)
        l_end = (curr_center[0] + math.ceil(l[0]), curr_center[1] + math.ceil(l[1]))

        # Mean displacement along x-axis
        u = curr_width//2 - l_end[0]
        # Mean displacement along y-axis
        v = curr_height//2 - l_end[1]

        # Track accumulated displacement
        self.cumulative_u_displacement += u
        self.cumulative_v_displacement += v

        str_err_u = f'Cumulative x displacement: {self.cumulative_u_displacement:.2f}'
        str_err_v = f'Cumulative y displacement: {self.cumulative_v_displacement:.2f}'

        # Draw mean flow and print errors onto image
        draw_image = cv2.arrowedLine(draw_image, curr_center, tuple(l_end), (0,0,255), 4)  
        draw_image = cv2.putText(draw_image, str_err_u, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235,0,255), 1)
        draw_image = cv2.putText(draw_image, str_err_v, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235,0,255), 1)
        
        # Publish data to topics
        self.flow_image.publish(self.bridge.cv2_to_imgmsg(draw_image, encoding="bgr8"))
        flow_errors = {
            "flow_error": self.cumulative_u_displacement
        }
        json_fe = json.dumps(flow_errors)
        self.flow_error.publish(json_fe)

        # Save current frame/features for next iteration
        self.lk_features = displaced_points.reshape(-1,1,2)
        self.optical_flow_loop_count += 1
        self.prev_frame = curr_frame
        
    #------------------------------------------------------------------------------------------------------
        
    def perform_camera_detection(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()
            if self.camera_image is None: continue

            # create a copy of the camera image
            image = self.camera_image.copy()
            # undistort camera image
            image = self.undistort_image(image)
            # get the clean and draw image
            clean_image = image.copy()

            draw_image = image

            # perform lane detection
            #draw_image = self.perform_simple_lane_detection(clean_image.copy(), draw_image)

            # peform colored tape detection
            #draw_image = self.perform_ground_color_detection(clean_image.copy(), draw_image)

            # perform optical flow computation
            draw_image = self.perform_optical_flow_detection(clean_image.copy(), clean_image.copy())
               
            # publish the image
            self.camera_detection_image_topic.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))

            # end the loop iteration
            rate.sleep()
            end_time = rospy.Time.now()
            duration = ((end_time-start_time)*1000).to_sec()
            rospy.loginfo(f"Loop duration: {duration:.2f} ms")


    def on_shutdown(self):
        # on shutdown
        print("terminate")

if __name__ == '__main__':
    node = CameraDetectionNode(node_name='camera_detection_node')
    rospy.sleep(2)
    node.perform_camera_detection()
    rospy.spin()
