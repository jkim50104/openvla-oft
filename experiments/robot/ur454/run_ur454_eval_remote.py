"""
run_ur454_eval.py

Evaluates a model in a real-world UR5 environment.
"""

import logging
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union
import cv2
from PIL import Image
import tensorflow_datasets as tfds
import numpy as np

import draccus
import tqdm

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.ur454.ur454_utils import (
    get_ur_env,
    get_ur_image,
    get_ur_wrist_images,
    get_next_task_label,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_from_server,
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
    get_model,
    get_action,
)

from prismatic.vla.constants import PROPRIO_DIM

sys.path.insert(0, "/home/jokim/projects/openvla-oft/third_party/Streaming-Grounded-SAM-2")
from grounded_sam2 import GroundedSam2Processor
# from third_party.Streaming-Grounded-SAM-2.grounded_sam2 import GroundedSam2Processor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    use_vla_server: bool = False                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = ""            # Remote VLA server URL (set to 127.0.0.1 if on same machine)

    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 10                    # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization


    #################################################################################################################
    # UR environment-specific parameters
    #################################################################################################################
    num_rollouts_planned: int = 50                   # Number of test rollouts
    max_steps: int = 1500                            # Max number of steps per rollout
    use_relative_actions: bool = True               # Whether to use relative actions (delta joint angles)

    host_ip: str = "192.168.0.3"
    speed: float = 1. #0.25
    acceleration: float = 1. #0.5

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.105, -0.372, 0.337])
    init_ee_rot: List[float] = field(default_factory=lambda: [-2.203, 2.235, -0.000])
    init_jpos: List[float] = field(default_factory=lambda: [1.060, -1.411, 1.898, -2.054, -1.564, 4.220])

    camera_topics = ["/cam1/cam1/color/image_raw",
                     "/cam2/cam2/color/image_raw"]
    control_frequency: float = 10

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    seed: int = 7                                    # Random Seed (for reproducibility)

    save_video: bool = False
    sanity_check: bool = False
    # fmt: on

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=PROPRIO_DIM,  # 7-dimensional proprio for UR
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Check that the model contains the action un-normalization key
    assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    # Create run ID
    run_id = f"EVAL-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id

def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_server_endpoint(cfg: GenerateConfig):
    """Get the server endpoint for remote inference."""
    ip_address = socket.gethostbyname(cfg.vla_server_url)
    return f"http://{ip_address}:8777/act"


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_ur_image(obs)
    wrist_img = get_ur_wrist_images(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": obs["state"]
    }

    return observation, img_resized, wrist_img_resized

def process_masks(masks, resize_size=224):
    return np.stack([
        np.asarray(
            Image.fromarray(mask).resize((resize_size, resize_size), Image.LANCZOS)
        ).astype(bool)
        for mask in masks
    ])


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    server_endpoint: str,
    resize_size,
    log_file=None,
    gs2=None,
):
    """Run a single episode in the UR environment."""
    # Define control frequency
    STEP_DURATION_IN_SEC = 1.0 / cfg.control_frequency

    # Reset environment
    obs = env.reset()

    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    curr_state = None
    replay_images = []
    replay_images_resized = []
    replay_images_wrist_resized = []

    if cfg.sanity_check:
        steps = iter(next(sanity_check_episode)["steps"])

    log_message("Prepare the scene, and then press Enter to begin...", log_file)
    input()

    # Fetch initial robot state (but sleep first so that robot stops moving)
    time.sleep(1)
    curr_state = env.get_tpos()

    episode_start_time = time.time()
    total_model_query_time = 0.0

    if gs2 is not None:
        gs2.extract_objects(task_description)
        idx = 0

    try:
        while t < cfg.max_steps:
            # Get step start time (used to compute how much to sleep between steps)
            step_start_time = time.time()

            # Get observation
            obs = env.get_observation()
            
            if gs2 is not None:
                seg_masks_info = gs2.segment_objects(cv2.cvtColor(obs["full_image"], cv2.COLOR_BGR2RGB), idx)
                idx+=1

            if cfg.sanity_check:
                obs["full_image"] = next(steps)["observation"]["image"].numpy()

            # print("[INFO] Curruent UR5 state:", obs["state"])
            cv2.imshow("Realsense View", np.hstack((obs["full_image"], obs["wrist_image"])))
            key = cv2.waitKey(1)

            if key == ord('q'):  # If 'q' is pressed, exit the loop
                env.close_env()  # Call your cleanup function
                break

            # Save raw high camera image for replay video
            replay_images.append(obs["full_image"])

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Prepare observation
                observation, img_resized, wrist_resized = prepare_observation(obs, resize_size)
                if gs2 is not None:
                    print(resize_size)       
                    seg_masks_info["masks"] = process_masks(seg_masks_info["masks"], resize_size)
                    observation["seg_masks_info"] = seg_masks_info
                observation["instruction"] = task_description

                # Save processed images for replay
                replay_images_resized.append(img_resized)
                replay_images_wrist_resized.append(wrist_resized)

                # Query model to get action
                log_message("Requerying model...", log_file)
                model_query_start_time = time.time()
                actions = get_action_from_server(observation, server_endpoint)
                actions = actions[: cfg.num_open_loop_steps]
                total_model_query_time += time.time() - model_query_start_time
                action_queue.extend(actions)
            
            # Get action from queue
            action = action_queue.popleft()
            log_message("-----------------------------------------------------", log_file)
            log_message(f"t: {t}", log_file)
            log_message(f"action: {action}", log_file)

            # Execute action in environment
            if cfg.use_relative_actions:
                # Get absolute joint angles from relative action
                target_state = curr_state + action[:6]
                action = np.concatenate([target_state, action[6:]])
                obs = env.step(action)
                # Update current state (assume it is the commanded target state)
                curr_state = target_state
            else:
                obs = env.step(action)
            t += 1

            # Sleep until next timestep
            step_elapsed_time = time.time() - step_start_time
            if step_elapsed_time < STEP_DURATION_IN_SEC:
                time_to_sleep = STEP_DURATION_IN_SEC - step_elapsed_time
                log_message(f"Sleeping {time_to_sleep} sec...", log_file)
                time.sleep(time_to_sleep)

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            log_message("\nCaught KeyboardInterrupt: Terminating episode early.", log_file)
        else:
            log_message(f"\nCaught exception: {e}", log_file)

    episode_end_time = time.time()

    # Get success feedback from user
    user_input = input("[Enter to terminate] Success? Enter 'y' or 'n': ")
    if user_input == "":
        env.close()
        exit()
    
    success = True if user_input.lower() == "y" else False

    # Calculate episode statistics
    episode_stats = {
        "success": success,
        "total_steps": t,
        "model_query_time": total_model_query_time,
        "episode_duration": episode_end_time - episode_start_time,
    }

    return (
        episode_stats,
        replay_images,
        replay_images_resized,
        replay_images_wrist_resized,
    )


def save_episode_videos(
    replay_images,
    replay_images_resized,
    replay_images_wrist,
    episode_idx,
    success,
    task_description,
    log_file=None,
):
    """Save videos of the episode from different camera angles."""
    # Save main replay video
    if success:
        save_rollout_video(
            replay_images, 
            episode_idx, 
            success=success, 
            task_description=task_description, 
            log_file=log_file)

    # # Save processed view videos
    # save_rollout_video(
    #     replay_images_resized,
    #     episode_idx,
    #     success=success,
    #     task_description=task_description,
    #     log_file=log_file,
    #     notes="resized",
    # )
    # save_rollout_video(
    #     replay_images_wrist,
    #     episode_idx,
    #     success=success,
    #     task_description=task_description,
    #     log_file=log_file,
    #     notes="wrist_resized",
    # )

@draccus.wrap()
def eval_ur454(cfg: GenerateConfig) -> None:
    """Main function to evaluate a trained policy in a real-world ALOHA environment."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Initialize the UR environment
    env = get_ur_env(cfg)
    
    # Initialize Grounded-SAM2 model
    gs2 = GroundedSam2Processor()
    
    # Get server endpoint for remote inference
    server_endpoint = get_server_endpoint(cfg)

    if cfg.sanity_check:
        ds = tfds.load(cfg.unnorm_key, split='train')
        sanity_check_episode = iter(ds.take(10))

    print("[INFO] Start evaluating in ur454")
    # Start evaluation
    # Pick up the bannana 
    task_description = "" #"put the banana to the yellow plate" #banana, strawberry

    num_rollouts_completed, total_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_rollouts_planned)):
        # Get task description from user
        task_description = get_next_task_label(task_description)

        if task_description == "q":
            print("[INFO] Finishing experiment early.")
            break

        log_message(f"\nTask: {task_description}", log_file)

        log_message(f"Starting episode {num_rollouts_completed + 1}...", log_file)

        # Run episode
        episode_stats, replay_images, replay_images_resized, replay_images_wrist = (
            run_episode(cfg, env, task_description, 
                        server_endpoint,
                        resize_size, log_file,
                        gs2)
        )

        # Update counters
        num_rollouts_completed += 1
        if episode_stats["success"]:
            total_successes += 1

        # Save videos
        save_episode_videos(
            replay_images,
            replay_images_resized,
            replay_images_wrist,
            num_rollouts_completed,
            episode_stats["success"],
            task_description,
            log_file,
        )

        # Log results
        log_message(f"Success: {episode_stats['success']}", log_file)
        log_message(f"# episodes completed so far: {num_rollouts_completed}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / num_rollouts_completed * 100:.1f}%)", log_file)
        log_message(f"Total model query time: {episode_stats['model_query_time']:.2f} sec", log_file)
        log_message(f"Total episode elapsed time: {episode_stats['episode_duration']:.2f} sec", log_file)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(num_rollouts_completed) if num_rollouts_completed > 0 else 0

    # Log final results
    log_message("\nFinal results:", log_file)
    log_message(f"Total episodes: {num_rollouts_completed}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    if log_file:
        log_file.close()

    # return final_success_rate
    env.close_env()

if __name__ == "__main__":
    eval_ur454()
