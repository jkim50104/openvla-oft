import time
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

import sys
sys.path.append('/home/jokim/projects/openvla-oft/experiments/robot/ur454')
from ur454_robot.robotiq_gripper_control import RobotiqGripper

import pyRobotiqGripper

# --- Main Control Script ---
def main():
    # UR IP
    ROBOT_IP = "192.168.0.3"

    # Connect to UR
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    gripper = RobotiqGripper(rtde_c, rtde_r)
    
    print(rtde_r.getActualQ())

    # Perform some gripper actions
    gripper.close()
    gripper.open()
    
    # Gripper range 0~50mm 0 is fully closed and 50 is fully opened
    # gripper.close()
    for i in  range(0, 50, 5):
        print("Move to:", i)
        gripper.move(i)
        pos = gripper.get_gripper_position()
        print(f"Gripper position (mm): {pos}")
    
    rtde_c.stopScript()


if __name__ == "__main__":
    main()
