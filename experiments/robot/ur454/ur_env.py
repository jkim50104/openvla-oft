"""
URGym environment definition.
"""

import time
from typing import Dict
import gym
import numpy as np

from ur454_robot.ur_env_service import URClient

def wait_for_obs(ur_client, timeout=5.0):
    start_time = time.time()
    obs = ur_client._get_observation()

    while obs is None:
        print("📷 Waiting for camera image...")
        time.sleep(0.1)
        obs = ur_client._get_observation()
        
        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError("❌ Timed out waiting for camera image.")

    return obs

class URGym(gym.Env):
    def __init__(
        self,
        ur_client: URClient,
        cfg: Dict,
        im_size: int = 256,
    ):
        self.ur_client = ur_client
        self.cfg = cfg
        self.im_size = im_size

        self.observation_space = gym.spaces.Dict({
            "full_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "wrist_image": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "proprio": gym.spaces.Box(low=np.ones((7,)) * -1, high=np.ones((8,)), dtype=np.float64),
        })

        self.action_space = gym.spaces.Box(low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64)

    def step(self, action):
        # Directly move using rtde
        self.ur_client.step_action(action)

        # obs = self.get_observation()
        obs = None

        return obs

    def reset(self, seed=None, options=None):
        #self.ur_client.reset()
        self.move_to_start_state()

        super().reset(seed=seed)

        obs = self.get_observation()

        return obs
    
    def get_observation(self):                
        return wait_for_obs(self.ur_client)
    
    def get_tpos(self):
        return self.ur_client._get_tcp_pos()
    
    def get_qpos(self):
        return self.ur_client._get_qpos()

    def move_to_start_state(self):
        print("🔁 Moving to start state...")
        successful = False
        while not successful:
            try:
                # self.ur_client.move(self.ur_client.start_state)
                self.ur_client.moveJ(self.ur_client.start_jpos)
                # self.ur_client.move_gripper(0)
                # self.ur_client.move_gripper(1)
                successful = True
            except Exception as e:
                print(e)
                
    def close_env(self):
        self.ur_client.close()