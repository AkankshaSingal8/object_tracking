#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time
from mavros_msgs.srv import CommandBool, SetMode
# import apriltag
import numpy as np




IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])


pt_star = np.array([364, 293, 276, 293,276, 205,364, 205])


def image_callback(msg):
    global pt_star
    bridge = CvBridge()
    try:
        # Convert to OpenCV image format
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        
    except CvBridgeError as e:
        print(e)

    cv_image = cv_img.copy()
    options = apriltag.DetectorOptions(families='tag36h11')
    detector = apriltag.Detector(options)
    img = CvBridge().imgmsg_to_cv2(cv_img, "mono8")
    result = detector.detect(img)

    if len(result)>0:
        for r in result:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            cv2.line(cv_image, ptA, (pt_star[0],pt_star[1]), (0, 255, 0), 2)
            cv2.line(cv_image, ptB, (pt_star[2],pt_star[3]), (0, 255, 0), 2)
            cv2.line(cv_image, ptC, (pt_star[4],pt_star[5]), (0, 255, 0), 2)
            cv2.line(cv_image, ptD, (pt_star[6],pt_star[7]), (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(cv_image, (cX, cY), 5, (0, 0, 255), -1)
            cv2.circle(cv_image, ptA, 3, (255, 0, 0), -1)
            cv2.circle(cv_image, ptB, 3, (255, 0, 0), -1)
            cv2.circle(cv_image, ptC, 3, (255, 0, 0), -1)
            cv2.circle(cv_image, ptD, 3, (255, 0, 0), -1)
            cv2.circle(cv_image, ((pt_star[0]+pt_star[2]+pt_star[4]+pt_star[6])/4,(pt_star[1]+pt_star[3]+pt_star[5]+pt_star[7])/4), 5, (0, 255, 0), -1)
            cv2.circle(cv_image, (pt_star[0],pt_star[1]), 3, (0, 255, 0), -1)
            cv2.circle(cv_image, (pt_star[2],pt_star[3]), 3, (0, 255, 0), -1)
            cv2.circle(cv_image, (pt_star[4],pt_star[5]), 3, (0, 255, 0), -1)
            cv2.circle(cv_image, (pt_star[6],pt_star[7]), 3, (0, 255, 0), -1)
    

if __name__ == '__main__':

    rospy.init_node('drone_raw_image_viewer', anonymous=True)
    # Subscribe to the raw image topic
    rospy.Subscriber("/cgo3_camera/image_raw", Image, image_callback)
    print("Subscribed")
    rospy.spin()