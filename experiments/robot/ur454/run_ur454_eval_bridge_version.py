"""
run_bridgev2_eval.py

Runs a model in a real-world 454 environment with UR robot.

Usage:
    # OpenVLA:
    python experiments/robot/ur454/run_ur454_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
import cv2
import tensorflow_datasets as tfds

import draccus

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.ur454.ur454_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_ur_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    unnorm_key: str = "bridge_orig" #"bridge_orig", "ur454_dataset" 

    #################################################################################################################
    # UR environment-specific parameters
    #################################################################################################################
    host_ip: str = "192.168.0.3"
    speed: float = 0.25
    acceleration: float =0.5

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.105, -0.372, 0.337])
    init_ee_rot: List[float] = field(default_factory=lambda: [-2.203, 2.235, -0.000])
    init_jpos: List[float] = field(default_factory=lambda: [1.489, -1.819, 2.176, -1.925, -1.561, 4.647])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics = "/camera/camera/color/image_raw"

    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 500 #60                                         # Max number of timesteps per episode
    control_frequency: float = 5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)
    save_video: bool = False
    sanity_check: bool = False
    # fmt: on


@draccus.wrap()
def eval_model_in_ur454_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    print("[INFO] Start evaluating in ur454")

    # # [OpenVLA] Set action un-normalization key
    # cfg.unnorm_key = "bridge_orig"

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize the UR environment
    env = get_ur_env(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    if cfg.sanity_check:
        ds = tfds.load(cfg.unnorm_key, split='train')
        sanity_check_episode = iter(ds.take(10))

    # Start evaluation
    # Pick up the bannana 
    task_label = "put the banana to the yellow plate" #banana, strawberry
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        if cfg.sanity_check:
            steps = iter(next(sanity_check_episode)["steps"])

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    obs = refresh_obs(obs, env)

                    if cfg.sanity_check:
                        obs["full_image"] = next(steps)["observation"]["image"].numpy()

                    # print("[INFO] Curruent UR5 state:", obs["state"])
                    cv2.imshow("Realsense View", obs["full_image"])
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):  # If 'q' is pressed, exit the loop
                        env.close_env()  # Call your cleanup function
                        break

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])

                    # Get preprocessed image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        obs,
                        task_label,
                        processor=processor,
                    )

                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["proprio"])
                        rollout_actions.append(action)

                    # Execute action
                    print("action:", action)
                    _, _, _, _, _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        if cfg.save_video:
            save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1
        
    env.close_env()

if __name__ == "__main__":
    eval_model_in_ur454_env()
