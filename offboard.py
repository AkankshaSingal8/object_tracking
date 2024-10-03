#! /usr/bin/env python3

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

from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow.python.keras.models import Functional
from tensorflow import keras
import numpy as np
from matplotlib.image import imread
import pandas as pd
from keras_models import generate_ncp_model

def generate_hidden_list(model: Functional, return_numpy: bool = True):
    """
    Generates a list of tensors that are used as the hidden state for the argument model when it is used in single-step
    mode. The batch dimension (0th dimension) is assumed to be 1 and any other dimensions (seq len dimensions) are
    assumed to be 0

    :param return_numpy: Whether to return output as numpy array. If false, returns as keras tensor
    :param model: Single step functional model to infer hidden states for
    :return: list of hidden states with 0 as value
    """
    constructor = np.zeros if return_numpy else tf.zeros
    hiddens = []
    print("Length of model input shape: ", len(model.input_shape))
    if len(model.input_shape)==1:
        lool = model.input_shape[0][1:]
    else:
        print("model input shape: ", model.input_shape)
        lool = model.input_shape[1:]
    print("lool: ", lool)
    for input_shape in lool:  # ignore 1st output, as is this control output
        hidden = []
        for i, shape in enumerate(input_shape):
            if shape is None:
                if i == 0:  # batch dim
                    hidden.append(1)
                    continue
                elif i == 1:  # seq len dim
                    hidden.append(0)
                    continue
                else:
                    print("Unable to infer hidden state shape. Leaving as none")
            hidden.append(shape)
        hiddens.append(constructor(hidden))
    return hiddens

current_state = State()
current_pose = PoseStamped()

MODEL_MODE = False
VEL = False
CV_IMAGE = None
INDEX = 0
image_sequences = []
vel_msg = Twist()
COUNT = 0

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])


tf.config.set_visible_devices([], 'GPU')

rospy.loginfo("current working dir")
rospy.loginfo(os.getcwd())
root = '/home/iiitd/catkin_ws/src/rover_tracking/scripts/retrain_150traj_woscheduler_seed22222_lr0.0001_trainloss0.00144_valloss0.00030_coreset.h5'
goal_image_file = '/home/iiitd/catkin_ws/src/rover_tracking/scripts/goal_image.png'
goal_image = cv2.imread(goal_image_file)

DEFAULT_NCP_SEED = 22222

batch_size = None
seq_len = 64
augmentation_params = None
no_norm_layer = False
single_step = True
model = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
model.load_weights(root)


hiddens = generate_hidden_list(model= model, return_numpy=True)


#with tf.device('/cpu:0'):
#    model = tf.keras.models.load_model(root)

def state_cb(msg):
    global current_state
    current_state = msg

def position_cb(pose):
    global current_pose
    current_pose = pose

def image_callback(msg):
    global CV_IMAGE
    bridge = CvBridge()
    try:
        # Convert to OpenCV image format
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_img = cv2.resize(cv_img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])) 
        
    except CvBridgeError as e:
        print(e)

    CV_IMAGE = cv_img
    # rospy.loginfo("got image")
    #save_dir = "./image_feed"
    #if not os.path.exists(save_dir):
    #	os.makedirs(save_dir)

    # Define the filename
    #filename = os.path.join(save_dir, "image_{}.png".format(time.time()))
    
    # Save the image
    #cv2.imwrite(filename, cv_img)
    # rospy.loginfo("Image saved as: {}".format(filename))


def model_vel():
    global CV_IMAGE
    global MODEL_MODE
    global VEL
    global INDEX
    global image_sequences
    global vel_msg
    global COUNT
    global hiddens
    global goal_image

    rospy.loginfo("Entering function to get pred")
    img = CV_IMAGE

    img = np.array(img)
    im_network = np.expand_dims(img, 0)
    
    difference = cv2.absdiff(img, goal_image)
    diff_image = np.array(difference)
    diff_image = np.expand_dims(diff_image, 0)

    # save_dir = "./vel_input"
    # if not os.path.exists(save_dir):
    #   os.makedirs(save_dir)

    # Define the filename
    # filename = os.path.join(save_dir, "image_{}.png".format(time.time()))
    
    # Save the image
    # cv2.imwrite(filename, img)

    # print(INDEX)
    # if INDEX == 0:
    #    image_sequences = np.stack([img] * 64)
        
    #     print(image_sequences.shape)
        
    # else:
        
    #     image_sequences = np.vstack((image_sequences[0][1:], [img]))
    #     print(image_sequences.shape)

    # image_sequences = np.expand_dims(image_sequences, axis=0)
    # INDEX += 1

    # with tf.device('/cpu:0'):
    #    preds = model.predict(image_sequences)
    #     print(preds[0][63])
    
    output = model.predict([diff_image, *hiddens])
    
    hiddens = output[1:]
    preds = output[0][0]
    vx, vy, vz, omega_z = preds[0], preds[1], preds[2], preds[3]

    # Set the velocities in the message
    vel_msg.linear.x = vx 
    vel_msg.linear.y = vy 
    vel_msg.linear.z = vz 
    vel_msg.angular.z = omega_z 
    
    px = current_pose.pose.position.x
    py = current_pose.pose.position.y
    pz = current_pose.pose.position.z

    #rospy.loginfo("got velocity")
    rospy.loginfo(f"Current Position: x={px}, y={py}, z={pz} ")
    rospy.loginfo(f" velocity: vx={vx}, vy={vy}, vz={vz}, omega_z={omega_z}")
    VEL = True
    COUNT += 1

    
if __name__ == "__main__":
    rospy.init_node("offb_node_py")

    state_sub = rospy.Subscriber("mavros/state", State, callback = state_cb)

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    local_position_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, position_cb)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    
    velocity_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)

    img_subscriber = rospy.Subscriber("/cgo3_camera/image_raw", Image, image_callback)


    # Setpoint publishing MUST be faster than 20Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    pose = PoseStamped()

    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 5

    # vel_msg = Twist()
    # vel_msg.linear.x = 0.5 
    # vel_msg.linear.y = 0
    # vel_msg.linear.z = 0
    # vel_msg.angular.z = 0

    # Send a few setpoints before starting
    for i in range(100):
        if(rospy.is_shutdown()):
            break

        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    while(not rospy.is_shutdown()):
        if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()
        else:
            if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")

                last_req = rospy.Time.now()
        
        
        

        if MODEL_MODE == False or VEL == False:
            # rospy.loginfo("Publishing pose")
            local_pos_pub.publish(pose)
            # print("pose", current_pose.pose.position.z)
            if current_pose.pose.position.z > 4.8:
                rospy.loginfo("Target pos reached")
                MODEL_MODE = True
        else:
            
            velocity_publisher.publish(vel_msg)
            rospy.loginfo("Vel published")
                
        
        if MODEL_MODE:
            rospy.loginfo("Calculating vel")
            model_vel()  
        

        rate.sleep()
