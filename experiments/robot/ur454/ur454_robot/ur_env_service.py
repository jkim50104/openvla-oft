import numpy as np
import cv2
from skimage.transform import resize
import threading
import rclpy
from rclpy.executors import SingleThreadedExecutor
import signal
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur454_robot.robotiq_gripper_control import RobotiqGripper
from ur454_robot.camera_subscriber import CameraSubscriber

class URClient():
    def __init__(self, host, env_params):
        self.host = host
        self.return_full_image = env_params.get("return_full_image", False)
        self.start_state = env_params.get("start_state")
        self.start_jpos = env_params.get("start_jpos")
        self.speed = env_params.get("speed", 0.25)
        self.acceleration = env_params.get("acceleration", 0.5)
        self.camera_topics = env_params["camera_topics"]

        self._initialize_robot()

    def _initialize_cam(self):
        # signal.signal(signal.SIGINT, signal.SIG_DFL)
        # if not self.cam_init:
        rclpy.init()
        self.executor = SingleThreadedExecutor()

        self.cam_nodes = [
            CameraSubscriber(topic=topic, node_name=f"camera_subscriber_{idx+1}")
            for idx, topic in enumerate(self.camera_topics)
        ]

        for node in self.cam_nodes:
            self.executor.add_node(node)

        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()

    def _spin_ros(self):
        while rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.01)
            except Exception as e:
                print(f"[ROS Spin Error]: {e}")
                for node in self.cam_nodes:
                    node.destroy_node()
                try:                
                    rclpy.shutdown()
                except:
                    print("[rclpy shutdown error]")
                print("🛑 Cleaning up ROS2 cam nodes...")

    def _initialize_robot(self):
        self.rtde_c = RTDEControlInterface(self.host)
        self.rtde_r = RTDEReceiveInterface(self.host)
        self.gripper = RobotiqGripper(self.rtde_c, self.rtde_r)
    
    def _get_processed_image(self, image=None, image_size=256):
        downsampled_trimmed_image = resize(image, (image_size, image_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        return downsampled_trimmed_image
    
    def _get_observation(self):
        image1 = self.cam_nodes[0].get_image()  # dtype = np.uint8
        image2 = self.cam_nodes[1].get_image()

        if image1 is None or image2 is None:
            return None

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        obs = {
            "full_image": image1,
            "wrist_image": image2,
            "state": np.concatenate([self._get_tcp_pos(), self._get_gripper_pos()]),  # [x,y,z,rx,ry,rz,gripper]
            # "state": np.concatenate([self._get_qpos(), self._get_gripper_pos()]) # [6D joints, gripper]
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
        return np.array([self.gripper.get_gripper_position() / self.gripper.max_range]) # gripper pose normalize
    
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
        print("🛑 Shutting down URClient and ROS2...")        
        self.rtde_c.stopScript()

        for node in self.cam_nodes:
            node.destroy_node()

        self.executor.shutdown()
        rclpy.shutdown()
        cv2.destroyAllWindows()