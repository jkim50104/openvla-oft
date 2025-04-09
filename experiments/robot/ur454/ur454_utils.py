"""Utils for evaluating policies in real-world ur454 environments."""

import os
import sys
import time

import imageio
import numpy as np
from PIL import Image
import tensorflow as tf
import torch


sys.path.append(".")
from experiments.robot.ur454.ur_env import URGym, URClient
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
UR_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_ur_env_params(cfg):
    env_params = {}
    env_params["override_workspace_boundaries"] = cfg.bounds
    env_params["camera_topics"] = cfg.camera_topics
    env_params["return_full_image"] = True

    start_state = np.concatenate([cfg.init_ee_pos, cfg.init_ee_rot])
    env_params["start_state"] = list(start_state)
    env_params["start_jpos"] = list(cfg.init_jpos)
    env_params["speed"] = cfg.speed
    env_params["acceleration"] = cfg.acceleration

    return env_params

def get_ur_env(cfg):
    """Get UR control environment."""
    # Set up the UR environment parameters
    env_params = get_ur_env_params(cfg)
    
    # Connect to UR
    ur_client = URClient(host=cfg.host_ip)
    ur_client.init(env_params)
    env = URGym(
        ur_client,
        cfg=cfg,
    )
    return env

def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    # else:
    #     user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
    #     if user_input == "":
    #         pass  # Do nothing -> Let task_label be the same
    #     else:
    #         task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_video(rollout_images, idx):
    """Saves an MP4 replay of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    mp4_path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.mp4"
    
    # Use the 'ffmpeg' plugin with proper fps setting
    video_writer = imageio.get_writer(mp4_path, fps=5, codec='libx264', format='FFMPEG')
    
    for img in rollout_images:
        video_writer.append_data(img)
    
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, notes=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    filetag = f"{rollout_dir}/{DATE_TIME}--openvla--episode={idx}--success={success}--task={processed_task_description}"
    if notes is not None:
        filetag += f"--{notes}"
    mp4_path = f"{filetag}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=25)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def resize_image_for_preprocessing(img):
    """
    Takes numpy array corresponding to a single image and resizes to 256x256, exactly as done
    in the ALOHA data preprocessing script, which is used before converting the dataset to RLDS.
    """
    ALOHA_PREPROCESS_SIZE = 256
    img = np.array(
        Image.fromarray(img).resize((ALOHA_PREPROCESS_SIZE, ALOHA_PREPROCESS_SIZE), resample=Image.LANCZOS)
    )  # BICUBIC is default; specify explicitly to make it clear
    return img


def get_ur_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["full_image"]
    img = resize_image_for_preprocessing(img)
    return img


def get_ur_wrist_images(obs):
    """Extracts both wrist camera images from observations and preprocesses them."""
    wrist_img = obs.observation["wrist_image"]
    wrist_img = resize_image_for_preprocessing(wrist_img)
    return wrist_img
