#!/usr/bin/env python3
import json
import math
import os

import cv2
import numpy as np

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    suc, prev = cap.read()

    theta = 40
    add_subtract = 1

    while True:
        suc, img = cap.read()
        
        #           0          1           2           3 
        corners = [[100, 100], [300, 125], [300, 225], [100, 250]]
        points = np.array(corners, np.int32)
        points = points.reshape((-1, 1, 2))


        """
        #------------------------------------------------------------
        # corners[1] and corners[0] are the top two points of the AprilTag ([3] and [2]?)
        a = abs(corners[1][1] - corners[0][1])
        b = abs(corners[1][0] - corners[0][0])
        theta = math.degrees(math.atan(a/b))

        # if theta is small enough, say it's 0
        if theta < 10:
            theta = 0
        #------------------------------------------------------------
        """

        if theta > 70 or theta < -70:
            add_subtract = add_subtract*-1
        theta += 5 * add_subtract

        colour = (255,100,255)
        if theta < 20 and theta > -20:
            colour = (0,255,0)

        # GET THETA FROM ANGLE BETWEEN TOP LEFT AND TOP RIGHT CORNERS:
        #a = abs(ToI.corners[2][1] - ToI.corners[1][1])
        #b = abs(ToI.corners[2][0] - ToI.corners[1][0])
        #theta = math.degrees(math.atan(a/b))

        h, w = img.shape[:2]

        # Error bar line points:  3--------1---------2
        point1 = (w//2, 25)
        point2 = (point1[0] + math.ceil(20 * math.cos(math.radians(theta))), point1[1] + math.ceil(20 * math.sin(math.radians(theta))))
        point3 = (point1[0] - math.ceil(20 * math.cos(math.radians(theta))), point1[1] - math.ceil(20 * math.sin(math.radians(theta))))


        img = cv2.line(img, point1, point2, colour, 4)
        img = cv2.line(img, point1, point3, colour, 4)
        img = cv2.putText(img, str(theta), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)


        #img = cv2.polylines(img, [points], True, (255, 100, 255), 5)
        #img = cv2.putText(img, str(theta), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 255), 2)
        #img = cv2.circle(img, (100,100), 4, (255,0,0), 4)
        #img = cv2.circle(img, (300,125), 4, (255,0,0), 4)

        cv2.imshow('img', img)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break