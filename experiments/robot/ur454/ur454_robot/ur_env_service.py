import numpy as np
import cv2
from skimage.transform import resize
import threading
import rclpy
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur454_robot.robotiq_gripper_control import RobotiqGripper
from ur454_robot.camera_subscriber import CameraSubscriber

class URClient():
    def __init__(self, host):        
        self.host = host

    def init(self, env_params):
        self.bounds = env_params["override_workspace_boundaries"]
        self.return_full_image = env_params["return_full_image"]
        self.start_state = env_params["start_state"]
        self.start_jpos = env_params["start_jpos"]

        # Initialize ROS 2
        rclpy.init()
        self.cam_node = CameraSubscriber(topic=env_params["camera_topics"])
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.cam_node)

        # Spin ROS in a background thread
        def ros_spin():
            while rclpy.ok():
                self.executor.spin_once(timeout_sec=0.01)

        self.ros_thread = threading.Thread(target=ros_spin, daemon=True)
        self.ros_thread.start()

        # Connect to UR
        self.rtde_c = RTDEControlInterface(self.host)
        self.rtde_r = RTDEReceiveInterface(self.host)
        self.speed = env_params["speed"]
        self.acceleration = env_params["acceleration"]

        self.gripper = RobotiqGripper(self.rtde_c, self.rtde_r)
    
    def _get_processed_image(self, image=None, image_size=256):
        downsampled_trimmed_image = resize(image, (image_size, image_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        return downsampled_trimmed_image
    
    def _get_observation(self):
        image = self.cam_node.get_image() #dtype = np.uint8
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        obs = {
            "full_image": image,
            "wrist_image": image,
            "state": np.concatenate(self._get_tcp_pos(), self._get_gripper_pos()), # [x,y,z,rx,ry,rz,gripper]
            # "state": np.concatenate(self._get_qpos(), self._get_gripper_pos()) # [6D joints, gripper]
        }

        return obs
        
    def _get_tcp_pos(self):
        # j_angles = self.rtde_r.getActualQ()
        # j_vel = self.rtde_r.getActualQd()
        # j_effort = self.rtde_r.getActualCurrent()
        # eep = self.rtde_r.getActualTCPPose()
        return np.array(self.rtde_r.getActualTCPPose())
    
    def _get_qpos(self):
        return np.array(self.rtde_r.getActualQ())
    
    def _get_gripper_pos(self):
        return np.array(self.gripper.get_gripper_position() / self.gripper.max_range) # gripper pose normalize
    
    def step_action(self, action):
        self.moveL(action[:6])  # Move to this new pose        
        self.move_gripper(action[6])
        
    def moveL(self, pose):
        self.rtde_c.moveL(pose, speed=self.speed , acceleration=self.acceleration)
    
    def moveJ(self, j_pos):
        self.rtde_c.moveJ(j_pos, speed=self.speed , acceleration=self.acceleration)
            
    def move_gripper(self, state: float):
        if state > 0.5:
            self.gripper.open()
        else:
            self.gripper.close()
            
    def close(self):
        print("🛑 Shutting down URCLient and ROS2...")        
        self.rtde_c.stopScript()
        self.cam_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()