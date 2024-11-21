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

def run_hector_initialisation_and_record(script_path, position, orientation, output_directory, ibvs_script):
    """
    Runs the Hector_Initialisation.py script with the given position and orientation values,
    waits for 15 seconds, records a ROS bag, and then runs the IBVS_Static.py script.

    Args:
        script_path (str): Path to the Hector_Initialisation.py script.
        position (tuple): A tuple of three floats representing the x, y, and z positions.
        orientation (tuple): A tuple of four floats representing the x, y, z, and w orientation.
        output_directory (str): Directory where the ROS bag should be recorded.
        ibvs_script (str): Path to the IBVS_Static.py script to be executed after rosbag recording.
    
    Returns:
        None
    """
    try:
        # Unpack position and orientation tuples
        position_x, position_y, position_z = position
        orientation_x, orientation_y, orientation_z, orientation_w = orientation
        
        # Run the Hector_Initialisation script as a background process
        hector_process = subprocess.Popen(
            [
                "python", script_path,
                str(position_x), str(position_y), str(position_z),
                str(orientation_x), str(orientation_y), str(orientation_z), str(orientation_w)
            ]
        )
        print("Hector_Initialisation.py script started successfully.")

        # Wait for 15 seconds
        print("Waiting for 15 sec")
        time.sleep(15)
        
        # Run the rosbag record command
        # subprocess.run(
        #     ["rosbag", "record", "-a", "--output-prefix", output_directory],
        #     check=True
        # )
        # print(f"rosbag recording started in directory: {output_directory}")
        
        # Run the IBVS_Static script
        subprocess.run(
            ["python", ibvs_script],
            check=True
        )
        print(f"IBVS_Static.py script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running a subprocess: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    


# Example usage
script_path = "Hector_Initialisation.py"
position = (1.4, 1.5, 6.0)
orientation = (0.0, 0.0, 1.0, 0.0)
ibvs_script = "IBVS_Static.py"
output_directory="quadrant2"

run_hector_initialisation_and_record(script_path, position, orientation, ibvs_script)
