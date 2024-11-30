#!/usr/bin/env python
import rospy
import sys
import cv2
import numpy as np
from std_msgs.msg import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
import time
import math
import tf
from math import cos, exp, pi, sin
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import apriltag
import rosbag
import subprocess
import time

def run_subprocess(model_script):
    try:
        # The command to run a Python script using python3
        command = ["python3", model_script]

        # Run the subprocess using Popen
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Capture the output and errors
        stdout, stderr = process.communicate()

        # Check if the subprocess exited with an error
        if process.returncode != 0:
            print("An error occurred. Subprocess stderr:", stderr)
            return None

        # Print the captured output
        print("Captured output from subprocess:", stdout.strip())

        return stdout.strip()  # Return the output to the caller
    except Exception as e:
        print("An unexpected error occurred:", e)
        return None

if __name__ == "__main__":
    # Specify the subprocess script name
    model_script = "model_trial.py"

    # Run the subprocess
    run_subprocess(model_script)

