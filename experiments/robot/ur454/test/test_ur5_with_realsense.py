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
from rclpy.executors import SingleThreadedExecutor

import sys
sys.path.append('/home/jokim/projects/openvla-oft/experiments/robot/ur454')
from ur454_robot.robotiq_gripper_control import RobotiqGripper

# --- ROS 2 Camera Subscriber Node ---
class CameraSubscriber(Node):
    def __init__(self, topic="/camera/camera/color/image_raw"):
        super().__init__('camera_subscriber')
        self.bridge = CvBridge()
        self.image = None
        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.image = cv_img
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def get_image(self):
        with self.lock:
            return self.image.copy() if self.image is not None else None


# --- Main Control Script ---
def main():
    # UR IP
    ROBOT_IP = "192.168.0.3"
    camera_topics = ["/cam1/cam1/color/image_raw",
                     "/cam2/cam2/color/image_raw"]

    # Initialize ROS 2
    rclpy.init()

    executor = SingleThreadedExecutor()
    cam_nodes = [CameraSubscriber(topic=topic) for topic in camera_topics]

    for node in cam_nodes:
        executor.add_node(node)

    # Spin ROS in a background thread
    def ros_spin():
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    # Connect to UR
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    gripper = RobotiqGripper(rtde_c, rtde_r)

    # Perform some gripper actions
    print(rtde_r.getActualQ())
    print(gripper.get_gripper_position())
    gripper.open()
    print(gripper.get_gripper_position())
    gripper.close()
    print(rtde_r.getActualQ())

    # Define a few target poses (x, y, z, Rx, Ry, Rz)
    poses = [
        [0.052, -0.413, 0.313, -0.479, 3.102, 0.003],
        [0.052, -0.413, 0.413, -0.479, 3.102, 0.003],
        [0.052, -0.413, 0.313, -0.479, 3.102, 0.003]
    ]

    print("Starting test loop. Press Ctrl+C to exit.")
    try:
        for idx, pose in enumerate(poses):
            print(f"\n[Step {idx+1}] Moving to: {pose}")
            rtde_c.moveL(pose, speed=0.25, acceleration=0.5)
            time.sleep(1)

            image1 = cam_nodes[0].get_image()  # dtype = np.uint8
            image2 = cam_nodes[1].get_image()
            if image1 is not None and image2 is not None:
                cv2.imshow("Realsense View", np.hstack((image1, image2)))
                cv2.waitKey(500)  # Display for 0.5 seconds
            else:
                print("⚠️ No image received yet.")

        print("\n✅ Done with all movements.")

    except (KeyboardInterrupt, Exception)  as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n🔴 Interrupted by user.")
        else:
            print(f"\nCaught exception: {e}")
        # exit(1)

    finally:
        rtde_c.stopScript()
        for cam_node in cam_nodes:
            cam_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Cleaning ROS")


if __name__ == "__main__":
    main()
