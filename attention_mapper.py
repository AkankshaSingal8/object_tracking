#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped


from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras
import numpy as np
from matplotlib.image import imread
import pandas as pd

from generate_vis_model import generate_conv_model, compute_visualbackprop, plot_saliency_map

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

tf.config.set_visible_devices([], 'GPU')

print(os.getcwd())

root = '/home/akanksha/catkin_ws/src/rover_tracking/scripts/model_fine_tuned.h5'
saliency_directory = '/home/akanksha/catkin_ws/src/rover_tracking/saliency_maps'
vis_model = generate_conv_model(root)

def image_callback(msg):
    global vis_model
    global saliency_directory
    if not os.path.exists(saliency_directory):
        os.makedirs(saliency_directory)

    print("entering_function")
    bridge = CvBridge()
    try:
        # Convert to OpenCV image format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = cv2.resize(cv_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])) 
       
        # print("converted and resized to cv img")
    except CvBridgeError as e:
        print(e)
    
    img_array = np.array(cv_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.convert_to_tensor(img_array)
    saliency_map = compute_visualbackprop(img, vis_model)
    file = "image_{}.png".format(time.time())
    plot_saliency_map(saliency_map, (224, 224), saliency_directory, file)

    save_dir = "/home/akanksha/catkin_ws/src/rover_tracking/image_feed"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the filename
    filename = os.path.join(save_dir, "image_{}.png".format(time.time()))
    
    # Save the image
    cv2.imwrite(filename, cv_image)
    rospy.loginfo("Image saved as: {}".format(filename))
    

def main():
    
    rospy.init_node('drone_raw_image_viewer', anonymous=True)
    # Subscribe to the raw image topic
    rospy.Subscriber("/cgo3_camera/image_raw", Image, image_callback)
    print("Subscribed")
    
    
    
    rospy.spin()
    

if __name__ == '__main__':
    main()
